#include "CudaSplitBadElement.h"
#include "CudaInsertPoint.h"
#include "CudaMesh.h"
#include <time.h>

// This function splits the encroached segments iteratively
void splitBadElements(
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
	IntD t_badeleidlist;
	IntD t_encseglist, t_encsubfacelist, t_badtetlist;
	IntD t_threadmarker;

	TetHandleD t_recordoldtetlist;
	IntD t_recordoldtetidx;

	int numberofbadeles;
	int numberofencsegs, numberofencsubfaces, numberofbadtets;

	clock_t tv[2];
	int npt[2];
	int code = 1;
	int iteration = 0;
	int counter;
	while (true)
	{
		//printf("%d, ", iteration);
		// Update the active bad elements list.
		// Exclude the empty ones (their encroachment markers have already been set to -1).
		numberofencsegs = updateActiveListByMarker_Slot(t_segencmarker, t_encseglist, numofsubseg);
		numberofencsubfaces = updateActiveListByMarker_Slot(t_subfaceencmarker, t_encsubfacelist, numofsubface);
		numberofbadtets = updateActiveListByStatus_Slot(t_tetstatus, t_badtetlist, numoftet);
		if (numberofbadtets == 0)
			break;
		if (numberofbadtets <= behavior->minbadtets && iteration >= behavior->miniter)
		{
			code = 0;
			break;
		}

		numberofbadeles = numberofencsegs + numberofencsubfaces + numberofbadtets;
		if (debug_msg) printf("      Iteration #%d: number of bad elements = %d (#%d segs, #%d subfaces, #%d tets)\n", 
			iteration, numberofbadeles, numberofencsegs, numberofencsubfaces, numberofbadtets);

		t_badeleidlist.resize(numberofbadeles);
		thrust::copy_n(t_encseglist.begin(), numberofencsegs, t_badeleidlist.begin());
		thrust::copy_n(t_encsubfacelist.begin(), numberofencsubfaces, t_badeleidlist.begin() + numberofencsegs);
		thrust::copy_n(t_badtetlist.begin(), numberofbadtets, t_badeleidlist.begin() + numberofencsegs + numberofencsubfaces);

		t_threadmarker.resize(numberofbadeles);
		thrust::fill_n(t_threadmarker.begin(), numberofencsegs, 0);
		thrust::fill_n(t_threadmarker.begin() + numberofencsegs, numberofencsubfaces, 1);
		thrust::fill_n(t_threadmarker.begin() + numberofencsegs + numberofencsubfaces, numberofbadtets, 2);

		//tv[0] = clock();
		//npt[0] = numofpoints;
		// Insert points concurrently
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
				t_badeleidlist,
				t_threadmarker,
				t_recordoldtetlist,
				t_recordoldtetidx,
				numberofbadeles,
				numberofencsegs,
				numberofencsubfaces,
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

		if (!code)
			break;

		cudaDeviceSynchronize();

		//if (iteration == 3)
		//	break;

		iteration++;

		//tv[1] = clock();
		//npt[1] = numofpoints;
		//printf("%f, %d\n", (REAL)(tv[1] - tv[0]), npt[1] - npt[0]);
	}
	if (!code && debug_msg)
		printf("      End with %d bad tets\n", numberofbadtets);
}