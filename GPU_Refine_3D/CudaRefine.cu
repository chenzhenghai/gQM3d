#include <stdio.h>
#include "CudaRefine.h"
#include "MeshReconstruct.h"
#include "CudaMesh.h"
#include "CudaThrust.h"
#include "CudaSplitEncseg.h"
#include "CudaSplitEncsubface.h"
#include "CudaSplitBadtet.h"
#include "CudaSplitBadElement.h"
#include "CudaCompactMesh.h"

/*************************************************************************************/
/*																					 */
/*  GPU_Refine_3D()   Compute 3D constrained Delaunay refinement on GPU.	         */
/*                                                                                   */
/*************************************************************************************/

void GPU_Refine_3D(
	MESHIO* input_gpu,
	MESHBH* input_behavior,
	int& out_numofpoint,
	double*& out_pointlist,
	int& out_numofedge,
	int*& out_edgelist,
	int& out_numoftriface,
	int*& out_trifacelist,
	int& out_numoftet,
	int*& out_tetlist
)
{
	/* Check input behavior */
	if (input_behavior->vecmode != 1 && input_behavior->vecmode != 2)
	{
		printf("Unknown vectorization mode: #%d\n", input_behavior->vecmode);
		exit(0);
	}

	internalmesh* drawmesh = input_behavior->drawmesh;

	/* Set up timer */
	StopWatchInterface *inner_timer = 0;
	sdkCreateTimer(&inner_timer);

	/******************************************/
	/* 0. Reconstruct the input cdt mesh      */
	/******************************************/
	printf("   0. Reconstructing the input CDT mesh...\n");
	
	// Reset and start timer.
	sdkResetTimer( &inner_timer );
	sdkStartTimer( &inner_timer );

	// input variables
	tethandle* inpoint2tetlist;
	trihandle* inpoint2trilist;
	verttype* inpointtypelist;
	int innumofedge;
	int* inseglist;
	trihandle* inseg2trilist;
	tethandle* inseg2tetlist;
	int innumoftriface;
	int* intrifacelist;
	tethandle* intri2tetlist;
	trihandle* intri2trilist;
	trihandle* intri2seglist;
	int innumoftetrahedron;
	int* intetlist;
	tethandle* inneighborlist;
	trihandle* intet2trilist;
	trihandle* intet2seglist;

	// reconstruct mesh
	reconstructMesh(
		input_gpu,
		inpoint2tetlist,
		inpoint2trilist,
		inpointtypelist,
		innumofedge,
		inseglist,
		inseg2trilist,
		inseg2tetlist,
		innumoftriface,
		intrifacelist,
		intri2tetlist,
		intri2trilist,
		intri2seglist,
		innumoftetrahedron,
		intetlist,
		inneighborlist,
		intet2trilist,
		intet2seglist,
		false
	);

	// Construct segment to parent list
	int* inseg2parentidxlist;
	int* insegparentendpointidxlist;
	int innumofsegparent;
	makesegment2parentmap(
		innumofedge,
		inseglist,
		inseg2trilist,
		inseg2parentidxlist,
		insegparentendpointidxlist,
		innumofsegparent);

	// Construct subface endpoint list
	// Although there are only triangles in the input PLC, the parents of subfaces may still be
	// polygon because Tetgen merged the nearly-coplaner
	int* intri2parentidxlist;
	int* inid2triparentoffsetlist;
	int* intriparentendpointidxlist;
	int innumoftriparent;
	int innumoftriparentendpoint;
	makesubfacepointsmap(
		input_gpu->numofpoints,
		input_gpu->pointlist,
		inpointtypelist,
		innumoftriface,
		intrifacelist,
		intri2seglist,
		intri2trilist,
		intri2parentidxlist,
		inid2triparentoffsetlist,
		intriparentendpointidxlist,
		innumoftriparent,
		innumoftriparentendpoint
	);

	// stop timer
	sdkStopTimer(&inner_timer);

	// print out info
	printf("      Reconstructed Mesh Size:\n");
	printf("        Number of point = %d\n", input_gpu->numofpoints);
	printf("        Number of edge = %d\n", innumofedge);
	printf("        Number of triface = %d\n", innumoftriface);
	printf("        Number of tetrahedron = %d\n", innumoftetrahedron);
	printf("      Reconstruction time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	input_behavior->times[0] = sdkGetTimerValue(&inner_timer);

	/******************************************/
	/* 1. Initialization				      */
	/******************************************/
	printf("   1. Initialization\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// Control variables
	int last_point = input_gpu->numofpoints;
	int last_subseg = innumofedge;
	int last_subface = innumoftriface;
	int last_subfaceparent = innumoftriparent;
	int last_tet = innumoftetrahedron;

	// Input mesh arrays, copy from the host
	RealD t_pointlist(input_gpu->pointlist, input_gpu->pointlist + 3 * last_point);
	TetHandleD t_point2tetlist(inpoint2tetlist, inpoint2tetlist + last_point);
	TriHandleD t_point2trilist(inpoint2trilist, inpoint2trilist + last_point);
	PointTypeD t_pointtypelist(inpointtypelist, inpointtypelist + last_point);
	IntD t_seglist(inseglist, inseglist + 3 * last_subseg);
	TriHandleD t_seg2trilist(inseg2trilist, inseg2trilist + 3 * last_subseg);
	TetHandleD t_seg2tetlist(inseg2tetlist, inseg2tetlist + last_subseg);
	IntD t_seg2parentidxlist(inseg2parentidxlist, inseg2parentidxlist + last_subseg);
	IntD t_segparentendpointidxlist(insegparentendpointidxlist, insegparentendpointidxlist + 2 * innumofsegparent);
	IntD t_trifacelist(intrifacelist, intrifacelist + 3 * last_subface);
	TetHandleD t_tri2tetlist(intri2tetlist, intri2tetlist + 2 * last_subface);
	TriHandleD t_tri2trilist(intri2trilist, intri2trilist + 3 * last_subface);
	TriHandleD t_tri2seglist(intri2seglist, intri2seglist + 3 * last_subface);
	IntD t_tri2parentidxlist(intri2parentidxlist, intri2parentidxlist + last_subface);
	IntD t_triid2parentoffsetlist(inid2triparentoffsetlist, inid2triparentoffsetlist + last_subfaceparent + 1);
	IntD t_triparentendpointidxlist(intriparentendpointidxlist, intriparentendpointidxlist + innumoftriparentendpoint);
	IntD t_tetlist(intetlist, intetlist + 4 * last_tet);
	TetHandleD t_neighborlist(inneighborlist, inneighborlist + 4 * last_tet);
	TriHandleD t_tet2trilist(intet2trilist, intet2trilist + 4 * last_tet);
	TriHandleD t_tet2seglist(intet2seglist, intet2seglist + 6 * last_tet);

	// Internal arrays
	RealD t_pointradius(last_point, 0.0);
	if (input_gpu->interpointradius != NULL)
		thrust::copy(input_gpu->interpointradius, input_gpu->interpointradius + last_point, t_pointradius.begin());

	TetStatusD t_tetstatus(last_tet, tetstatus(1));
	TriStatusD t_tristatus(last_subface, tristatus(1));
	TriStatusD t_segstatus(last_subseg, tristatus(1));

	// Marker arrays
	IntD t_segencmarker(last_subseg, -1); // initialize to non-encroached
	IntD t_subfaceencmarker(last_subface, -1);

	// Cuda mesh manipulation
	int xmax, xmin, ymax, ymin, zmax, zmin;
	cudamesh_inittables();
	cudamesh_initbbox(input_gpu->numofpoints, input_gpu->pointlist,
		xmax, xmin, ymax, ymin, zmax, zmin);
	cudamesh_exactinit(0, 0, 0, xmax - xmin, ymax - ymin, zmax - zmin);
	cudamesh_initkernelconstants(xmax - xmin, ymax - ymin, zmax - zmin);

	// initialize subseg encroach marker
	initSegEncmarkers(
		t_pointlist,
		t_seglist,
		t_seg2tetlist,
		t_segencmarker,
		t_tetlist,
		t_neighborlist,
		last_subseg
	);

	initSubfaceEncmarkers(
		t_pointlist,
		t_trifacelist,
		t_tri2tetlist,
		t_subfaceencmarker,
		t_tetlist,
		last_subface
	);

	initTetBadstatus(
		t_pointlist,
		t_tetlist,
		t_tetstatus,
		input_behavior->radius_to_edge_ratio,
		last_tet
	);

	// stop timer
	sdkStopTimer(&inner_timer);

	// print out info
	printf("      Initialization time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	input_behavior->times[1] = sdkGetTimerValue(&inner_timer);

	//gpuMemoryCheck();

	/******************************************/
	/* 2. Split encroached subsegments		  */
	/******************************************/
	printf("   2. Split encroached subsegments\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// split encroached segments
	splitEncsegs(
		t_pointlist,
		t_point2trilist,
		t_point2tetlist,
		t_pointtypelist,
		t_pointradius,
		t_seglist,
		t_seg2trilist,
		t_seg2tetlist,
		t_seg2parentidxlist,
		t_segparentendpointidxlist,
		t_segstatus,
		t_trifacelist,
		t_tri2tetlist,
		t_tri2trilist,
		t_tri2seglist,
		t_tri2parentidxlist,
		t_triid2parentoffsetlist,
		t_triparentendpointidxlist,
		t_tristatus,
		t_tetlist,
		t_neighborlist,
		t_tet2trilist,
		t_tet2seglist,
		t_tetstatus,
		t_segencmarker,
		t_subfaceencmarker,
		last_point,
		last_subseg,
		last_subface,
		last_tet,
		input_behavior,
		-1,
		-1,
		0,
		false,
		false
	);

	// stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);

	// print out info
	printf("      Splitting encroached subsegments time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	printf("      Number of points = %d, segments = %d, subfaces = %d, tets = %d\n",
		last_point, last_subseg, last_subface, last_tet);
	input_behavior->times[2] = sdkGetTimerValue(&inner_timer);

	if (drawmesh != NULL && !drawmesh->animation)
	{
		if (drawmesh->iter_seg != -1 && drawmesh->iter_subface == -1 && drawmesh->iter_tet == -1)
			return;
	}

	/******************************************/
	/* 3. Split encroached subfaces		      */
	/******************************************/

	printf("   3. Split encroached subfaces\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	// split encroached subfaces
	splitEncsubfaces(
		t_pointlist,
		t_point2trilist,
		t_point2tetlist,
		t_pointtypelist,
		t_pointradius,
		t_seglist,
		t_seg2trilist,
		t_seg2tetlist,
		t_seg2parentidxlist,
		t_segparentendpointidxlist,
		t_segstatus,
		t_trifacelist,
		t_tri2tetlist,
		t_tri2trilist,
		t_tri2seglist,
		t_tri2parentidxlist,
		t_triid2parentoffsetlist,
		t_triparentendpointidxlist,
		t_tristatus,
		t_tetlist,
		t_neighborlist,
		t_tet2trilist,
		t_tet2seglist,
		t_tetstatus,
		t_segencmarker,
		t_subfaceencmarker,
		last_point,
		last_subseg,
		last_subface,
		last_tet,
		input_behavior,
		-1,
		0,
		false,
		false
	);

	// stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);

	// print out info
	printf("      Splitting encroached subfaces time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
	printf("      Number of points = %d, segments = %d, subfaces = %d, tets = %d\n",
		last_point, last_subseg, last_subface, last_tet);
	input_behavior->times[3] = sdkGetTimerValue(&inner_timer);

	if (drawmesh != NULL && !drawmesh->animation)
	{
		if (drawmesh->iter_subface != -1 && drawmesh->iter_tet == -1)
			return;
	}

	if (input_behavior->vecmode == 2)
	{

		/******************************************/
		/* 4. Split bad elements        		  */
		/******************************************/

		printf("   4. Split bad elements\n");

		// Reset and start timer.
		sdkResetTimer(&inner_timer);
		sdkStartTimer(&inner_timer);

		// split encroached segments
		splitBadElements(
			t_pointlist,
			t_point2trilist,
			t_point2tetlist,
			t_pointtypelist,
			t_pointradius,
			t_seglist,
			t_seg2trilist,
			t_seg2tetlist,
			t_seg2parentidxlist,
			t_segparentendpointidxlist,
			t_segstatus,
			t_trifacelist,
			t_tri2tetlist,
			t_tri2trilist,
			t_tri2seglist,
			t_tri2parentidxlist,
			t_triid2parentoffsetlist,
			t_triparentendpointidxlist,
			t_tristatus,
			t_tetlist,
			t_neighborlist,
			t_tet2trilist,
			t_tet2seglist,
			t_tetstatus,
			t_segencmarker,
			t_subfaceencmarker,
			last_point,
			last_subseg,
			last_subface,
			last_tet,
			input_behavior,
			1,
			false,
			false
		);

		// stop timer
		cudaDeviceSynchronize();
		sdkStopTimer(&inner_timer);

		// print out info
		printf("      Splitting bad elements time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
		printf("      Number of points = %d, segments = %d, subfaces = %d, tets = %d\n",
			last_point, last_subseg, last_subface, last_tet);
		input_behavior->times[4] = sdkGetTimerValue(&inner_timer);

		if (drawmesh != NULL && !drawmesh->animation)
		{
			if (drawmesh->iter_tet != -1)
				return;
		}
	}
	else if (input_behavior->vecmode == 1)
	{
		/******************************************/
		/* 4. Split bad quality tets		      */
		/******************************************/

		//gpuMemoryCheck();

		printf("   4. Split bad quality tets\n");

		// Reset and start timer.
		sdkResetTimer(&inner_timer);
		sdkStartTimer(&inner_timer);

		// split bad tets
		splitBadTets(
			t_pointlist,
			t_point2trilist,
			t_point2tetlist,
			t_pointtypelist,
			t_pointradius,
			t_seglist,
			t_seg2trilist,
			t_seg2tetlist,
			t_seg2parentidxlist,
			t_segparentendpointidxlist,
			t_segstatus,
			t_trifacelist,
			t_tri2tetlist,
			t_tri2trilist,
			t_tri2seglist,
			t_tri2parentidxlist,
			t_triid2parentoffsetlist,
			t_triparentendpointidxlist,
			t_tristatus,
			t_tetlist,
			t_neighborlist,
			t_tet2trilist,
			t_tet2seglist,
			t_tetstatus,
			t_segencmarker,
			t_subfaceencmarker,
			last_point,
			last_subseg,
			last_subface,
			last_tet,
			input_behavior,
			1,
			false,
			false
		);

		// stop timer
		cudaDeviceSynchronize();
		sdkStopTimer(&inner_timer);

		// print out info
		printf("      Splitting bad tets time = %.3f ms\n", sdkGetTimerValue(&inner_timer));
		printf("      Number of points = %d, segments = %d, subfaces = %d, tets = %d\n",
			last_point, last_subseg, last_subface, last_tet);
		input_behavior->times[4] = sdkGetTimerValue(&inner_timer);

		if (drawmesh != NULL && !drawmesh->animation)
		{
			if (drawmesh->iter_tet != -1)
				return;
		}
	}

	/******************************************/
	/* 5. Output final quality mesh		      */
	/******************************************/

	printf("   5. Output final quality mesh\n");

	// Reset and start timer.
	sdkResetTimer(&inner_timer);
	sdkStartTimer(&inner_timer);

	compactMesh(
		out_numofpoint, out_pointlist, t_pointlist,
		out_numofedge, out_edgelist, t_seglist, t_segstatus,
		out_numoftriface, out_trifacelist, t_trifacelist, t_tristatus,
		out_numoftet, out_tetlist, t_tetlist, t_tetstatus
		);

	// stop timer
	cudaDeviceSynchronize();
	sdkStopTimer(&inner_timer);
	input_behavior->times[5] = sdkGetTimerValue(&inner_timer);
}