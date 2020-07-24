#include "CudaSplitEncseg.h"
#include "CudaSplitEncsubface.h"
#include "CudaSplitBadtet.h"
#include "CudaInsertPoint.h"
#include "CudaMesh.h"
#include <time.h>

/* Host */
// This function assumes the input status has be set correctly
// in the initialization
void initTetBadstatus(
	RealD& t_pointlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	REAL minratio,
	int& numofbadtet
)
{
	int numberofblocks = (ceil)((float)numofbadtet / BLOCK_SIZE);
	kernelMarkAllBadtets << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		minratio,
		numofbadtet
		);
}

// This function splits the bad tets iteratively
void splitBadTets(
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
	int debug_msg,
	bool debug_error,
	bool debug_timing
)
{
	int numberofbadtets; // number of bad tets
	IntD t_badtetlist;
	IntD t_threadmarker;

	TetHandleD t_recordoldtetlist;
	IntD t_recordoldtetidx;

	clock_t tv[2];
	int npt[2];
	int code = 1;
	int iteration = 0;
	int counter = 0;
	while (true)
	{
		// Update the active bad tet list.
		// Exclude the empty ones (their status have already been set to empty).
		numberofbadtets = updateActiveListByStatus_Slot(t_tetstatus, t_badtetlist, numoftet);
		if (debug_msg) printf("      Iteration #%d: number of bad tets = %d\n", iteration, numberofbadtets);
		if (numberofbadtets == 0)
			break;

		if (numberofbadtets <= behavior->minbadtets && iteration >= behavior->miniter)
		{
			code = 0;
			break;
		}


		t_threadmarker.resize(numberofbadtets);
		thrust::fill(t_threadmarker.begin(), t_threadmarker.end(), 2);

		//cudaDeviceSynchronize();
		//tv[0] = clock();
		//npt[0] = numofpoints;

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
				t_badtetlist,
				t_threadmarker,
				t_recordoldtetlist,
				t_recordoldtetidx,
				numberofbadtets,
				0,
				0, // split tets
				numberofbadtets,
				numofpoints,
				numofsubseg,
				numofsubface,
				numoftet,
				behavior,
				-1,
				-1,
				iteration,
				debug_msg,
				debug_error,
				debug_timing
			);

		//cudaDeviceSynchronize();
		//tv[1] = clock();
		//npt[1] = numofpoints;
		//printf("%f, %d\n", (REAL)(tv[1] - tv[0]), npt[1] - npt[0]);

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
			-1,
			iteration,
			0,
			debug_error,
			false
		);

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
			numofpoints,
			numofsubseg,
			numofsubface,
			numoftet,
			behavior,
			iteration,
			0,
			debug_error,
			false
		);

		//cudaDeviceSynchronize();
		//tv[0] = clock();
		//npt[0] = numofpoints;
		//printf("%f, %d\n", (REAL)(tv[0] - tv[1]), npt[0] - npt[1]);

		iteration++;
	}

	if (!code)
		printf("      Ended with %d bad tets\n", numberofbadtets);
}