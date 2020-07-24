#include "CudaSplitEncseg.h"
#include "CudaInsertPoint.h"
#include "CudaMesh.h"

/* Host */
// This function assumes the input encmarker has be set correctly
// in the initialization
void initSegEncmarkers(
	RealD& t_pointlist,
	IntD& t_seglist,
	TetHandleD& t_seg2tetlist,
	IntD& t_segencmarker,
	IntD& t_tetlist,
	TetHandleD& t_neighborlist,
	int& numofsubseg
)
{
	int numberofblocks = (ceil)((float)numofsubseg / BLOCK_SIZE);
	kernelMarkAllEncsegs<<<numberofblocks, BLOCK_SIZE>>> (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_segencmarker[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		numofsubseg
		);
}

// This function splits the encroached segments iteratively
void splitEncsegs(
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
	int iter_subface,
	int iter_tet,
	int debug_msg,
	bool debug_error,
	bool debug_timing
)
{
	int numberofencsegs; // number of encroached subsegs
	IntD t_encseglist;
	IntD t_threadmarker;

	TetHandleD t_recordoldtetlist;
	IntD t_recordoldtetidx;

	int code = 1;
	int iteration = 0;
	int counter;
	while (true)
	{

		// Update the active encroached segment list.
		// Exclude the empty ones (their markers have already been set to -1).
		numberofencsegs = updateActiveListByMarker_Slot(t_segencmarker, t_encseglist, numofsubseg);
		if(debug_msg) printf("      Iteration #%d: number of encroached segments = %d\n", iteration, numberofencsegs);
		if (numberofencsegs == 0)
			break;

		t_threadmarker.resize(numberofencsegs);
		thrust::fill(t_threadmarker.begin(), t_threadmarker.end(), 0);

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
				t_encseglist,
				t_threadmarker,
				t_recordoldtetlist,
				t_recordoldtetidx,
				numberofencsegs,
				numberofencsegs,
				0,
				0,
				numofpoints,
				numofsubseg,
				numofsubface,
				numoftet,
				behavior,
				iteration,
				iter_subface,
				iter_tet,
				debug_msg,
				debug_error,
				debug_timing
			);

		if (!code)
			break;

		cudaDeviceSynchronize();

		iteration++;
	}

	if (!code && debug_msg)
		printf("      Ended with %d bad segment\n", numberofencsegs);
}