#include "CudaSplitEncseg.h"
#include "CudaSplitEncsubface.h"
#include "CudaInsertPoint.h"
#include "CudaMesh.h"

/* Host */
// This function assumes the input encmarker has be set correctly
// in the initialization
void initSubfaceEncmarkers(
	RealD& t_pointlist,
	IntD& t_trifacelist,
	TetHandleD& t_tri2tetlist,
	IntD& t_subfaceencmarker,
	IntD& t_tetlist,
	int& numofsubface
)
{
	int numberofblocks = (ceil)((float)numofsubface / BLOCK_SIZE);
	kernelMarkAllEncsubfaces << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		numofsubface
		);
}

// This function splits the encroached subfaces iteratively
void splitEncsubfaces(
	RealD& t_pointlist,
	TriHandleD& t_point2trilist,
	TetHandleD& t_point2tetlist,
	PointTypeD& t_pointtypelist,
	RealD& t_pointradius,
	IntD& t_seglist,
	TriHandleD& t_seg2trilist,
	TetHandleD& t_seg2tetlist,
	IntD& t_seg2parentidxlist,
	IntD& t_segparentendpointidxlist,
	TriStatusD& t_segstatus,
	IntD& t_trifacelist,
	TetHandleD& t_tri2tetlist,
	TriHandleD& t_tri2trilist,
	TriHandleD& t_tri2seglist,
	IntD& t_tri2parentidxlist,
	IntD& t_triid2parentoffsetlist,
	IntD& t_triparentendpointidxlist,
	TriStatusD& t_tristatus,
	IntD& t_tetlist,
	TetHandleD& t_neighborlist,
	TriHandleD& t_tet2trilist,
	TriHandleD& t_tet2seglist,
	TetStatusD& t_tetstatus,
	IntD& t_segencmarker,
	IntD& t_subfaceencmarker,
	int& numofpoints,
	int& numofsubseg,
	int& numofsubface,
	int& numoftet,
	MESHBH* behavior,
	int iter_tet,
	int debug_msg,
	bool debug_error,
	bool debug_timing
)
{
	int numberofencsubfaces; // number of encroached subfaces
	IntD t_encsubfacelist;
	IntD t_threadmarker;

	TetHandleD t_recordoldtetlist;
	IntD t_recordoldtetidx;

	int code = 1;
	int iteration = 0;
	while (true)
	{

		// Update the active encroached subface list.
		// Exclude the empty ones (their markers have already been set to -1).
		numberofencsubfaces = updateActiveListByMarker_Slot(t_subfaceencmarker, t_encsubfacelist, numofsubface);
		if(debug_msg) printf("      Iteration #%d: number of encroached subfaces = %d\n", iteration, numberofencsubfaces);
		if (numberofencsubfaces == 0)
			break;

		t_threadmarker.resize(numberofencsubfaces);
		thrust::fill(t_threadmarker.begin(), t_threadmarker.end(), 1);

		code = 
			insertPoint_New(
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
				t_encsubfacelist,
				t_threadmarker,
				t_recordoldtetlist,
				t_recordoldtetidx,
				numberofencsubfaces,
				0,
				numberofencsubfaces,
				0, // split subface
				numofpoints,
				numofsubseg,
				numofsubface,
				numoftet,
				behavior,
				-1,
				iteration,
				iter_tet,
				debug_msg,
				debug_error,
				debug_timing
			);

		if (!code)
			break;

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
			numofpoints,
			numofsubseg,
			numofsubface,
			numoftet,
			behavior,
			iteration,
			iter_tet,
			0,
			debug_error,
			false
		);

		cudaDeviceSynchronize();

		iteration++;
	}

	if (!code && debug_msg)
		printf("      Ended with %d bad subface\n", numberofencsubfaces);
}