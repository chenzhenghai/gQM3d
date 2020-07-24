#include "CudaInsertPoint.h"
#include "CudaSplitEncseg.h"
#include "CudaMesh.h"
#include "CudaAnimation.h"
#include <math_constants.h>
#include <time.h>

int insertPoint_New(
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
	IntD& t_insertidxlist,
	IntD& t_threadmarker, // indicate insertion type: -1 failed, 0 splitsubseg, 1 splitsubface, 2 splittet
	TetHandleD& t_recordoldtetlist,
	IntD& t_recordoldtetidx,
	int numofbadelements,
	int numofencsubseg,
	int numofencsubface,
	int numofbadtet,
	int& numofpoints,
	int& numofsubseg,
	int& numofsubface,
	int& numoftet,
	MESHBH* behavior,
	int iter_0, // iteration number for segment splitting or bad element splitting
	int iter_1, // iteration number for subface splitting
	int iter_2, // iteration number for tet splitting
	int debug_msg, // Debug only, output debug information
	bool debug_error,
	bool debug_timing
)
{
	clock_t tv[2];

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
	}

	internalmesh* drawmesh = behavior->drawmesh;
	int numofinsertpt = numofbadelements;

	// Initialization
	int numberofthreads;
	int numberofblocks;
	int numberofsplittablesubsegs;
	int numberofsplittablesubfaces;
	int numberofsplittabletets;

	IntD t_threadlist; // active thread list
	UInt64D t_tetmarker(numoftet, 0); // marker for tets. Used for cavity.
	IntD t_segmarker(numofsubseg, 0); // marker for subsegs. Used for splitting segments
	UInt64D t_trimarker(numofsubface, 0); // marker for subfaces. Used for subcavity.

	RealD t_insertptlist(3 * numofinsertpt);
	IntD t_priority(numofinsertpt, 0);
	RealD t_priorityreal(numofinsertpt, 0.0); // store real temporarily

	// Compute priorities
	numberofblocks = (ceil)((float)numofinsertpt / BLOCK_SIZE);
	kernelComputePriorities << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_priorityreal[0]),
		numofinsertpt
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}


	// Sort element indices by priorites and
	// pick the first N elements where N = behavior->maxbadelements
	if (behavior->filtermode == 2 && behavior->maxbadelements > 0 && numofbadelements > behavior->maxbadelements)
	{
		if (behavior->filterstatus == 1)
			behavior->filterstatus = 2;

		int numberofloser;
		if(numofbadtet > numofencsubseg && numofbadtet > numofencsubface)
		{
			numberofloser = numofbadtet - behavior->maxbadelements;
			if (numberofloser > 0)
			{
				thrust::sort_by_key(t_insertidxlist.begin() + numofencsubseg + numofencsubface, t_insertidxlist.end(),
					t_priorityreal.begin() + numofencsubseg + numofencsubface);
				thrust::fill(t_threadmarker.begin() + numofencsubseg + numofencsubface, 
					t_threadmarker.begin() + numofencsubseg + numofencsubface + numberofloser, -1);
			}
		}
	}
	else
	{
		if (behavior->filterstatus == 2)
			behavior->filterstatus = 3;
	}

	kernelComputeSteinerPoints << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_point2trilist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2parentidxlist[0]),
		thrust::raw_pointer_cast(&t_segparentendpointidxlist[0]),
		thrust::raw_pointer_cast(&t_segencmarker[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		numofinsertpt
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Modify priorities and convert them into integers
	// subsegment > subface > tet (smaller in values)
	double priority_min[3], priority_max[3], priority_offset[3] = { 0, 0, 0 };
	thrust::pair<RealD::iterator, RealD::iterator> priority_pair;
	if (numofbadtet > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin() + numofencsubseg + numofencsubface,
				t_priorityreal.end());
		priority_min[2] = *priority_pair.first;
		priority_max[2] = *priority_pair.second;
		priority_offset[2] = 0;
		if (debug_error)
		{
			printf("MinMax Real priorities for tet: %lf, %lf\n", priority_min[2], priority_max[2]);
			printf("Offset: %lf\n", priority_offset[2]);
		}
	}
	if (numofencsubface > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin() + numofencsubseg,
				t_priorityreal.begin() + numofencsubseg + numofencsubface);
		priority_min[1] = *priority_pair.first;
		priority_max[1] = *priority_pair.second;
		if (numofbadtet > 0)
			priority_offset[1] = priority_max[2] + priority_offset[2] + 10 - priority_min[1];
		else
			priority_offset[1] = 0;
		if (debug_error)
		{
			printf("MinMax Real priorities for subface: %lf, %lf\n", priority_min[1], priority_max[1]);
			printf("Offset: %lf\n", priority_offset[1]);
		}

	}
	if (numofencsubseg > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_priorityreal.begin(),
				t_priorityreal.begin() + numofencsubseg);
		priority_min[0] = *priority_pair.first;
		priority_max[0] = *priority_pair.second;
		if (numofencsubface > 0)
			priority_offset[0] = priority_max[1] + priority_offset[1] + 10 - priority_min[0];
		else if (numofbadtet > 0)
			priority_offset[0] = priority_max[2] + priority_offset[2] + 10 - priority_min[0];
		else
			priority_offset[0] = 0;
		if (debug_error)
		{
			printf("MinMax Real priorities for subseg: %lf, %lf\n", priority_min[0], priority_max[0]);
			printf("Offset: %lf\n", priority_offset[0]);
		}

	}
	
	kernelModifyPriority << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_priorityreal[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		priority_offset[0],
		priority_offset[1],
		priority_offset[2],
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numofinsertpt
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (debug_error)
	{

		if (numofencsubseg > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin(),
					t_priorityreal.begin() + numofencsubseg);
			priority_min[0] = *priority_pair.first;
			priority_max[0] = *priority_pair.second;
			printf("MinMax Real priorities for subseg: %lf, %lf\n", priority_min[0], priority_max[0]);
		}

		if (numofencsubface > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin() + numofencsubseg,
					t_priorityreal.begin() + numofencsubseg + numofencsubface);
			priority_min[1] = *priority_pair.first;
			priority_max[1] = *priority_pair.second;
			printf("MinMax Real priorities for subface: %lf, %lf\n", priority_min[1], priority_max[1]);
		}

		if (numofbadtet > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin() + numofencsubseg + numofencsubface,
					t_priorityreal.end());
			priority_min[2] = *priority_pair.first;
			priority_max[2] = *priority_pair.second;
			printf("MinMax Real priorities for tet: %lf, %lf\n", priority_min[2], priority_max[2]);
		}

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	freeVec(t_priorityreal);

	// To Add: Grid filtering

	// Reject points when violate insertradius rules
	kernelCheckInsertRadius << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_pointradius[0]),
		thrust::raw_pointer_cast(&t_point2trilist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_seg2parentidxlist[0]),
		thrust::raw_pointer_cast(&t_segparentendpointidxlist[0]),
		thrust::raw_pointer_cast(&t_segencmarker[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_tri2parentidxlist[0]),
		thrust::raw_pointer_cast(&t_triid2parentoffsetlist[0]),
		thrust::raw_pointer_cast(&t_triparentendpointidxlist[0]),
		thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		numofinsertpt
		);

	if (debug_error)
	{
		//int numofunsplittabletets = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isAbortiveTet());
		//printf("number of unsplittable tets = %d\n", numofunsplittabletets);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update working thread list
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (numberofthreads == 0)
		return 0;

	// verify number of splittable tets
	if (iter_0 == -1 && iter_1 == -1 && iter_2 >= 0)
	{
		numberofsplittabletets = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2);
		if (numberofsplittabletets == 0)
			return 0;
		if (numberofsplittabletets < behavior->minsplittabletets)
			return 0;
	}

	if (debug_msg >= 1)
		printf("        After insertradius check, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));

	// Locate Steiner points
	numberofsplittablesubsegs = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0);	
	thrust::device_vector<locateresult> t_pointlocation(numofinsertpt, UNKNOWN);
	TetHandleD t_searchtet(numofinsertpt, tethandle(-1, 11));
	TriHandleD t_searchsh(numofencsubseg + numofencsubface, trihandle(-1, 11));

	kernelLocatePoint << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tri2trilist[0]),
		thrust::raw_pointer_cast(&t_tri2seglist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_searchsh[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		numberofsplittablesubsegs,
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// update working thread list
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (debug_msg >= 2)
		printf("        After point location, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
		return 0;

	if (drawmesh != NULL && drawmesh->animation)
	{
		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
			iter_2 == drawmesh->iter_tet)
			outputStartingFrame(
				drawmesh,
				t_pointlist,
				t_tetlist,
				t_tetstatus,
				t_threadlist,
				t_insertidxlist,
				t_insertptlist,
				t_searchtet,
				iter_0,
				iter_1,
				iter_2
			);
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Initialization, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	TetHandleD t_caveoldtetlist; // list to record interior tets
	IntD t_caveoldtetidx;
	TetHandleD t_cavetetlist; // list to record tets in expanding cavities
	IntD t_cavetetidx;
	TriHandleD t_caveshlist; // list to record subfaces in subcavities
	IntD t_caveshidx;
	TriHandleD t_cavesegshlist; // list to record subface-at-splitedges
	IntD t_cavesegshidx;
	TetHandleD t_cavebdrylist; // list to record boundary tets
	IntD t_cavebdryidx;

	// Adatively reserve memory space
	// size and fac would fluctuate
	if (behavior->caveoldtetsizefac > 3.0)
		behavior->caveoldtetsizefac = 1.5;
	if (behavior->cavetetsizefac > 3.0)
		behavior->cavetetsizefac = 1.5;
	if (behavior->cavebdrysizefac > 3.0)
		behavior->cavebdrysizefac = 1.5;

	if (behavior->caveshsizefac > 3.0)
		behavior->caveshsizefac = 1.5;

	int resoldtetsize = behavior->caveoldtetsize * behavior->caveoldtetsizefac;
	t_caveoldtetlist.reserve(resoldtetsize);
	t_caveoldtetidx.reserve(resoldtetsize);
	int restetsize = behavior->cavetetsize * behavior->cavetetsizefac;
	t_cavetetlist.reserve(restetsize);
	t_cavetetidx.reserve(restetsize);
	int resbdrysize = behavior->cavebdrysize * behavior->cavebdrysizefac;
	t_cavebdrylist.reserve(resbdrysize);
	t_cavebdryidx.reserve(resbdrysize);

	int resshsize = behavior->caveshsize * behavior->caveshsizefac;
	t_caveshlist.reserve(resshsize);
	t_caveshidx.reserve(resshsize);

	// Compute initial cavity starting points
	int oldsize, newsize;
	int initialcavitysize, initialsubcavitysize;
	IntD t_initialcavitysize(numofinsertpt, MAXINT);
	IntD t_initialsubcavitysize(numofinsertpt, MAXINT);
	IntD t_initialcavityindices(numofinsertpt, -1);
	IntD t_initialsubcavityindices(numofinsertpt, -1);

	// set losers' cavity and scavity sizes to zero
	thrust::replace_if(t_initialcavitysize.begin(), t_initialcavitysize.end(), t_threadmarker.begin(), isNegativeInt(), 0);
	thrust::replace_if(t_initialsubcavitysize.begin(), t_initialsubcavitysize.end(), t_threadmarker.begin(), isNegativeInt(), 0);

	// Form initial cavities
	// mark and count the initial cavities
	// mark tets using original thread indices
	kernelMarkAndCountInitialCavity << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_searchsh[0]),
		thrust::raw_pointer_cast(&t_seg2trilist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2trilist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_segmarker[0]),
		thrust::raw_pointer_cast(&t_trimarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_initialcavitysize[0]),
		thrust::raw_pointer_cast(&t_initialsubcavitysize[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Check record oldtet lists
	if (behavior->cavitymode == 2)
	{
		numberofthreads = t_recordoldtetidx.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelCheckRecordOldtet << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_recordoldtetlist[0]),
				thrust::raw_pointer_cast(&t_recordoldtetidx[0]),
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_insertptlist[0]),
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_segstatus[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				thrust::raw_pointer_cast(&t_priority[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				thrust::raw_pointer_cast(&t_initialcavitysize[0]),
				thrust::raw_pointer_cast(&t_initialsubcavitysize[0]),
				numofencsubseg,
				numofencsubface,
				numofbadelements,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			kernelKeepRecordOldtet << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_recordoldtetidx[0]),
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}
	}

	// update working thread list
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	if (debug_msg >= 2)
		printf("        After initial cavity marking, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
	{
		// This should not error
		printf("Error: 0 threads after marking initial cavities!\n");
		exit(0);
	}

	// compute total size and indices for intital cavities
	thrust::exclusive_scan(t_initialcavitysize.begin(), t_initialcavitysize.end(), t_initialcavityindices.begin());
	thrust::exclusive_scan(t_initialsubcavitysize.begin(), t_initialsubcavitysize.end(), t_initialsubcavityindices.begin());
	initialcavitysize = t_initialcavityindices[numofinsertpt - 1] + t_initialcavitysize[numofinsertpt - 1];
	initialsubcavitysize = t_initialsubcavityindices[numofinsertpt - 1] + t_initialsubcavitysize[numofinsertpt - 1];
	if (debug_error)
	{
		printf("Initial cavity size = %d, subcavity size = %d\n", initialcavitysize, initialsubcavitysize);
	}

	// init cavity lists
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	t_caveoldtetlist.resize(initialcavitysize);
	t_caveoldtetidx.resize(initialcavitysize);

	int expandfactor = 4;
	t_cavetetlist.resize(expandfactor * initialcavitysize);
	t_cavetetidx.resize(expandfactor * initialcavitysize);
	thrust::fill(t_cavetetidx.begin(), t_cavetetidx.end(), -1); // some slots might not be used

	if (initialsubcavitysize > 0)
	{
		t_caveshlist.resize(initialsubcavitysize);
		t_caveshidx.resize(initialsubcavitysize);
		t_cavesegshlist.resize(initialsubcavitysize);
		t_cavesegshidx.resize(initialsubcavitysize);
		thrust::fill(t_cavesegshidx.begin(), t_cavesegshidx.end(), -1); // some slots might not be used
	}

	kernelInitCavityLinklist << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_searchsh[0]),
		thrust::raw_pointer_cast(&t_seg2trilist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tri2trilist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_initialcavityindices[0]),
		thrust::raw_pointer_cast(&t_initialcavitysize[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_cavetetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetidx[0]),
		thrust::raw_pointer_cast(&t_initialsubcavityindices[0]),
		thrust::raw_pointer_cast(&t_initialsubcavitysize[0]),
		thrust::raw_pointer_cast(&t_caveshlist[0]),
		thrust::raw_pointer_cast(&t_caveshidx[0]),
		thrust::raw_pointer_cast(&t_cavesegshlist[0]),
		thrust::raw_pointer_cast(&t_cavesegshidx[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	freeVec(t_initialcavitysize);
	freeVec(t_initialcavityindices);
	//freeVec(t_initialsubcavitysize);
	freeVec(t_initialsubcavityindices);

	if (behavior->cavitymode == 2)
	{
		auto first_record_iter = thrust::make_zip_iterator(thrust::make_tuple(t_recordoldtetlist.begin(), t_recordoldtetidx.begin()));
		auto last_record_iter = thrust::make_zip_iterator(thrust::make_tuple(t_recordoldtetlist.end(), t_recordoldtetidx.end()));

		int expandreusesize = thrust::count_if(t_recordoldtetidx.begin(), t_recordoldtetidx.end(), isTetIndexToReuse());
		//printf("expandreusesize = %d\n", expandreusesize);

		if (expandreusesize > 0)
		{
			// copy recordoldtet to oldtet
			int oldlistsize = t_caveoldtetlist.size();
			t_caveoldtetlist.resize(oldlistsize + expandreusesize);
			t_caveoldtetidx.resize(oldlistsize + expandreusesize);
			auto first_old_iter =
				thrust::make_zip_iterator(thrust::make_tuple(t_caveoldtetlist.begin() + oldlistsize, t_caveoldtetidx.begin() + oldlistsize));
			auto last_old_iter =
				thrust::copy_if(first_record_iter, last_record_iter, first_old_iter, isCavityTupleToReuse());
			//printf("distance = %d\n", thrust::distance(first_old_iter, last_old_iter));

			numberofthreads = expandreusesize; // each thread works on one tet in cavetetlist
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

			kernelSetReuseOldtet << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
				thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
				oldlistsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

#ifdef GQM3D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// expand cavetet
			IntD t_cavetetexpandsize(numberofthreads, 0), t_cavetetexpandindices(numberofthreads, -1);
			int cavetetexpandsize;

			kernelCheckCavetetFromReuseOldtet << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
				thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				oldlistsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			thrust::exclusive_scan(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), t_cavetetexpandindices.begin());
			cavetetexpandsize = t_cavetetexpandindices[numberofthreads - 1] + t_cavetetexpandsize[numberofthreads - 1];
			//printf("cavetetexpandsize = %d\n", cavetetexpandsize);
			int oldcavetetsize = t_cavetetlist.size();
			t_cavetetlist.resize(oldcavetetsize + cavetetexpandsize);
			t_cavetetidx.resize(oldcavetetsize + cavetetexpandsize);

			kernelAppendCavetetFromReuseOldtet << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
				thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
				thrust::raw_pointer_cast(&t_cavetetlist[0]),
				thrust::raw_pointer_cast(&t_cavetetidx[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				oldlistsize,
				oldcavetetsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		// remove used recordoldtet
		//printf("before remove: t_recordoldtet size = %d\n", t_recordoldtetlist.size());
		auto last_record_iter_remove = thrust::remove_if(first_record_iter, last_record_iter, isInvalidCavityTuple());
		int newlistsize = thrust::distance(first_record_iter, last_record_iter_remove);
		t_recordoldtetlist.resize(newlistsize);
		t_recordoldtetidx.resize(newlistsize);
		//printf("After remove: t_recordoldtet size = %d\n", t_recordoldtetlist.size());
	}
	
	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Init cavity lists, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Expand Initial Cavity
	// Every iteration, test if current tet in cavetetlist is included in cavity
	// If it is, expand cavetetlist and caveoldtetlist, otherwise expand cavebdrylist
	int cavetetcurstartindex = 0;
	int cavetetstartindex = t_cavetetlist.size();
	int caveoldtetstartindex = t_caveoldtetlist.size();
	int cavebdrystartindex = t_cavebdrylist.size();
	int cavetetexpandsize = cavetetstartindex, caveoldtetexpandsize, cavebdryexpandsize;

	numberofthreads = cavetetexpandsize; // each thread works on one tet in cavetetlist
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	IntD t_cavetetexpandsize(numberofthreads, 0);
	IntD t_caveoldtetexpandsize(numberofthreads, 0);
	IntD t_cavebdryexpandsize(numberofthreads, 0);
	IntD t_cavetetexpandindices(numberofthreads, -1);
	IntD t_caveoldtetexpandindices(numberofthreads, -1);
	IntD t_cavebdryexpandindices(numberofthreads, -1);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
	}
	int iteration = 0;
	while (true)
	{
		if (behavior->cavitymode == 1 && iteration > behavior->maxcavity) // Too large cavities. Stop and mark as unsplittable elements
		{
			kernelLargeCavityCheck << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_insertptlist[0]),
				thrust::raw_pointer_cast(&t_cavetetidx[0]),
				thrust::raw_pointer_cast(&t_segstatus[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				cavetetcurstartindex,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			break;
		}
		else if (behavior->cavitymode == 2 && iteration > behavior->mincavity)
		{
			int oldnumofthreads = numberofthreads;

			kernelMarkCavityReuse << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_cavetetidx[0]),
				thrust::raw_pointer_cast(&t_segstatus[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				cavetetcurstartindex,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			numberofthreads = t_caveoldtetlist.size();
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelMarkOldtetlist << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
				thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			int expandrecordsize = thrust::count_if(t_caveoldtetlist.begin(), t_caveoldtetlist.end(), isInvalidTetHandle());
			//printf("expandrecordsize = %d\n", expandrecordsize);
			int oldrecordsize = t_recordoldtetidx.size();
			t_recordoldtetlist.resize(oldrecordsize + expandrecordsize);
			t_recordoldtetidx.resize(oldrecordsize + expandrecordsize);
			auto first_old_iter = thrust::make_zip_iterator(thrust::make_tuple(t_caveoldtetlist.begin(), t_caveoldtetidx.begin()));
			auto last_old_iter = thrust::make_zip_iterator(thrust::make_tuple(t_caveoldtetlist.end(), t_caveoldtetidx.end()));
			auto first_record_iter =
				thrust::make_zip_iterator(
					thrust::make_tuple(
						t_recordoldtetlist.begin() + oldrecordsize,
						t_recordoldtetidx.begin() + oldrecordsize));
			auto last_record_iter =
				thrust::copy_if(first_old_iter, last_old_iter, first_record_iter, isCavityTupleToRecord());
			//printf("distance = %d\n", thrust::distance(first_record_iter, last_record_iter));

			numberofthreads = expandrecordsize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelSetRecordOldtet << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_recordoldtetlist[0]),
				thrust::raw_pointer_cast(&t_recordoldtetidx[0]),
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				oldrecordsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			numberofthreads = oldnumofthreads;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelMarkLargeCavityAsLoser << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_cavetetidx[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				cavetetcurstartindex,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			break;
		}

		// Check if current tet is included in cavity
		kernelCavityExpandingCheck << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavetetidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			thrust::raw_pointer_cast(&t_priority[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			cavetetcurstartindex,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		kernelCorrectExpandingSize << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_cavetetidx[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			cavetetcurstartindex,
			numberofthreads
			);


		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		thrust::exclusive_scan(
			thrust::make_zip_iterator(thrust::make_tuple(t_cavetetexpandsize.begin(), t_caveoldtetexpandsize.begin(), t_cavebdryexpandsize.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(t_cavetetexpandsize.end(), t_caveoldtetexpandsize.end(), t_cavebdryexpandsize.end())),
			thrust::make_zip_iterator(thrust::make_tuple(t_cavetetexpandindices.begin(), t_caveoldtetexpandindices.begin(), t_cavebdryexpandindices.begin())),
			thrust::make_tuple(0, 0, 0),
			PrefixSumTupleOP());

		// Count expanding sizes
		cavetetexpandsize = t_cavetetexpandindices[numberofthreads - 1] + t_cavetetexpandsize[numberofthreads - 1];
		caveoldtetexpandsize = t_caveoldtetexpandindices[numberofthreads - 1] + t_caveoldtetexpandsize[numberofthreads - 1];
		cavebdryexpandsize = t_cavebdryexpandindices[numberofthreads - 1] + t_cavebdryexpandsize[numberofthreads - 1];

		//if (debug_error)
		//{
		//	printf("Iteration = %d, cavetetexpandsize = %d, caveoldtetexpandsize = %d, cavebdryexpandsize = %d\n",
		//		iteration, cavetetexpandsize, caveoldtetexpandsize, cavebdryexpandsize);
		//}

		// Prepare memeory
		oldsize = t_cavetetlist.size();
		newsize = oldsize + cavetetexpandsize;
		t_cavetetlist.resize(newsize);
		t_cavetetidx.resize(newsize);
		oldsize = t_caveoldtetlist.size();
		newsize = oldsize + caveoldtetexpandsize;
		t_caveoldtetlist.resize(newsize);
		t_caveoldtetidx.resize(newsize);
		oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryidx.resize(newsize);

		kernelCavityExpandingMarkAndAppend << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavetetidx[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			cavetetstartindex,
			thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
			thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			caveoldtetstartindex,
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			cavebdrystartindex,
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			cavetetcurstartindex,
			numberofthreads
			);


		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		// Update working thread list
		numberofthreads = cavetetexpandsize;
		iteration++;
		if (numberofthreads == 0)
			break;

		// Update variables
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		cavetetcurstartindex = cavetetstartindex;
		cavetetstartindex = t_cavetetlist.size();
		caveoldtetstartindex = t_caveoldtetlist.size();
		cavebdrystartindex = t_cavebdrylist.size();

		// Reset expanding lists
		t_cavetetexpandsize.resize(numberofthreads);
		thrust::fill(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), 0);
		t_cavetetexpandindices.resize(numberofthreads);

		t_caveoldtetexpandsize.resize(numberofthreads);
		thrust::fill(t_caveoldtetexpandsize.begin(), t_caveoldtetexpandsize.end(), 0);
		t_caveoldtetexpandindices.resize(numberofthreads);

		t_cavebdryexpandsize.resize(numberofthreads);
		thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);
		t_cavebdryexpandindices.resize(numberofthreads);
	}

	// Update cavetet, caveoldtet, cavebdry sizes and factors
	behavior->cavetetsizefac = t_cavetetlist.size() * 1.0 / behavior->cavetetsize + 0.02;
	behavior->cavetetsize = t_cavetetlist.size();

	behavior->caveoldtetsizefac = t_caveoldtetlist.size() * 1.0 / behavior->caveoldtetsize + 0.02;
	behavior->caveoldtetsize = t_caveoldtetlist.size();

	if (behavior->filterstatus == 3)
	{
		behavior->cavetetsizefac = 1.1;
		behavior->caveoldtetsizefac = 1.1;
	}

	freeVec(t_cavetetlist);
	freeVec(t_cavetetidx);
	freeVec(t_cavetetexpandsize);
	freeVec(t_caveoldtetexpandsize);
	freeVec(t_cavetetexpandindices);
	freeVec(t_caveoldtetexpandindices);

	// Update working threadlist to winners
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (debug_msg >= 1)
		printf("        After expanding cavity, numberofthreads = %d(#%d, #%d, #%d), total expanding iteration = %d\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2),
			iteration);
	if (numberofthreads == 0)
		return 1;


	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Cavity expanding, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	numberofthreads = t_caveoldtetlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	// Collect segments in cavities
	TriHandleD t_cavetetseglist;
	IntD t_cavetetsegidx;
	IntD t_cavetetsegsize(numberofthreads, 0), t_cavetetsegindices(numberofthreads, -1);
	UInt64D t_segmarker2(numofsubseg, 0);

	kernelMarkCavityAdjacentSubsegs << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	thrust::fill(t_segmarker2.begin(), t_segmarker2.end(), 0);

	kernelCountCavitySubsegs_Phase1 << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	kernelCountCavitySubsegs_Phase2 << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_cavetetsegsize[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	thrust::exclusive_scan(t_cavetetsegsize.begin(), t_cavetetsegsize.end(), t_cavetetsegindices.begin());
	int cavetetsegsize = t_cavetetsegindices[numberofthreads - 1] + t_cavetetsegsize[numberofthreads - 1];
	t_cavetetseglist.resize(cavetetsegsize);
	t_cavetetsegidx.resize(cavetetsegsize);

	if (debug_error)
		printf("cavetetsegsize = %d\n", cavetetsegsize);

	kernelAppendCavitySubsegs << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
		thrust::raw_pointer_cast(&t_cavetetsegsize[0]),
		thrust::raw_pointer_cast(&t_cavetetsegindices[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Segment encroachment test
	numberofthreads = cavetetsegsize;
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		t_cavetetsegsize.resize(cavetetsegsize); // indicate encroached: 0: no 1: yes
		thrust::fill(t_cavetetsegsize.begin(), t_cavetetsegsize.end(), 0);

		kernelCheckSegmentEncroachment_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_segencmarker[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			thrust::raw_pointer_cast(&t_cavetetseglist[0]),
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_cavetetsegsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		kernelCheckSegmentEncroachment_Phase2 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_cavetetsegsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			//int numofunsplittabletets = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isAbortiveTet());
			//printf("number of unsplittable tets = %d\n", numofunsplittabletets);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (debug_msg >= 1)
		printf("        After segment encroachment check, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
		return 1;

	freeVec(t_cavetetsegsize);
	freeVec(t_cavetetsegindices);
	freeVec(t_segmarker2);

	numberofthreads = t_caveoldtetlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	// Collect subfaces in cavities
	TriHandleD t_cavetetshlist;
	IntD t_cavetetshidx;
	IntD t_cavetetshsize(numberofthreads, 0), t_cavetetshindices(numberofthreads, -1);
	IntD t_trimarker2(numofsubface, 0);

	kernelMarkCavityAdjacentFaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	kernelCountCavitySubfaces_Phase1 << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_trimarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	kernelCountCavitySubfaces_Phase2 << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_cavetetshsize[0]),
		thrust::raw_pointer_cast(&t_trimarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	thrust::exclusive_scan(t_cavetetshsize.begin(), t_cavetetshsize.end(), t_cavetetshindices.begin());
	int cavetetshsize = t_cavetetshindices[numberofthreads - 1] + t_cavetetshsize[numberofthreads - 1];
	t_cavetetshlist.resize(cavetetshsize);
	t_cavetetshidx.resize(cavetetshsize);

	if (debug_error)
		printf("cavetetshsize = %d\n", cavetetshsize);

	kernelAppendCavitySubfaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshidx[0]),
		thrust::raw_pointer_cast(&t_cavetetshsize[0]),
		thrust::raw_pointer_cast(&t_cavetetshindices[0]),
		thrust::raw_pointer_cast(&t_trimarker2[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		//for (int i = 0; i < 100; i++)
		//{
		//	trihandle tmpsubface = t_cavetetshlist[i];
		//	int tmpidx = t_cavetetshidx[i];
		//	printf("%d(%d), %d\n", tmpsubface.id, tmpsubface.shver, tmpidx);
		//}
	}

	numberofthreads = cavetetshsize;
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		t_cavetetshsize.resize(cavetetshsize); // indicate encroached: 0: no 1: yes
		thrust::fill(t_cavetetshsize.begin(), t_cavetetshsize.end(), 0);

		kernelCheckSubfaceEncroachment_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_cavetetshlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_cavetetshsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		kernelCheckSubfaceEncroachment_Phase2 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_pointlocation[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_cavetetshsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			//int numofunsplittabletets = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isAbortiveTet());
			//printf("number of unsplittable tets = %d\n", numofunsplittabletets);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (debug_msg >= 1)
		printf("        After subface encroachment check, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));
	if (numberofthreads == 0)
		return 1;

	freeVec(t_cavetetshsize);
	freeVec(t_cavetetshindices);
	freeVec(t_trimarker2);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Subsegment and subface collection, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Expand subcavities
	if (t_caveshlist.size() != 0)
	{
		int caveshcurstartindex = 0;
		int caveshstartindex = t_caveshlist.size();
		int caveshexpandsize = caveshstartindex;

		numberofthreads = caveshexpandsize; // each thread works on one subface in caveshlist
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		IntD t_caveshexpandflag(numberofthreads, -1);
		IntD t_caveshexpandsize(numberofthreads, 0);
		IntD t_caveshexpandindices(numberofthreads, -1);

		iteration = 0;
		while (true)
		{
			kernelSubCavityExpandingCheck << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_insertptlist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_caveshexpandsize[0]),
				thrust::raw_pointer_cast(&t_caveshexpandflag[0]),
				thrust::raw_pointer_cast(&t_priority[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				caveshcurstartindex,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			thrust::exclusive_scan(t_caveshexpandsize.begin(), t_caveshexpandsize.end(), t_caveshexpandindices.begin());
			caveshexpandsize = t_caveshexpandindices[numberofthreads - 1] + t_caveshexpandsize[numberofthreads - 1];
			//if (debug_error)
			//	printf("iteration = %d, caveshexpandsize = %d\n", iteration, caveshexpandsize);
			iteration++;
			if (caveshexpandsize == 0)
				break;

			// Prepare memeory
			oldsize = t_caveshlist.size();
			newsize = oldsize + caveshexpandsize;
			t_caveshlist.resize(newsize);
			t_caveshidx.resize(newsize);

			// Append expanding subfaces
			kernelSubCavityExpandingAppend << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_caveshexpandsize[0]),
				thrust::raw_pointer_cast(&t_caveshexpandindices[0]),
				thrust::raw_pointer_cast(&t_caveshexpandflag[0]),
				caveshcurstartindex,
				caveshstartindex,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			// Update variables
			numberofthreads = caveshexpandsize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			caveshcurstartindex = caveshstartindex;
			caveshstartindex = t_caveshlist.size();

			// Reset expanding lists
			t_caveshexpandsize.resize(numberofthreads);
			thrust::fill(t_caveshexpandsize.begin(), t_caveshexpandsize.end(), 0);
			t_caveshexpandindices.resize(numberofthreads);
			t_caveshexpandflag.resize(numberofthreads);
			thrust::fill(t_caveshexpandflag.begin(), t_caveshexpandflag.end(), -1);
		}

		if (behavior->caveshsize == 0)
			behavior->caveshsizefac = 1.002;
		else
			behavior->caveshsizefac = t_caveshlist.size() * 1.0 / behavior->caveshsize + 0.02;
		behavior->caveshsize = t_caveshlist.size();

		if (behavior->filterstatus == 3)
			behavior->caveshsizefac = 1.1;

		if (debug_msg >= 2)
			printf("        Subcavity total expanding iteration = %d\n", iteration);
	}

	freeVec(t_priority);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Subcavity expanding, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Find boundary subfaces
	TetHandleD t_cavetetflag;
	numberofthreads = t_cavetetshlist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		t_cavebdryexpandsize.resize(numberofthreads);
		t_cavebdryexpandindices.resize(numberofthreads);
		thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);
		t_cavetetflag.resize(numberofthreads, tethandle(-1, 11));
		kernelCavityBoundarySubfacesCheck << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tet2trilist[0]),
			thrust::raw_pointer_cast(&t_tet2seglist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_cavetetflag[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());
		cavebdryexpandsize = t_cavebdryexpandindices[numberofthreads - 1] + t_cavebdryexpandsize[numberofthreads - 1];
		cavebdrystartindex = oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryidx.resize(newsize);
		if (debug_error)
			printf("cavebdryexpandsize = %d\n", cavebdryexpandsize);

		kernelCavityBoundarySubfacesAppend << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_cavetetflag[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			cavebdrystartindex,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	// Find boundary segments
	numberofthreads = t_cavetetseglist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		t_cavebdryexpandsize.resize(numberofthreads);
		t_cavebdryexpandindices.resize(numberofthreads);
		thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);
		t_cavetetflag.resize(numberofthreads);

		kernelCavityBoundarySubsegsCheck << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2tetlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavetetseglist[0]),
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_cavetetflag[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());
		cavebdryexpandsize = t_cavebdryexpandindices[numberofthreads - 1] + t_cavebdryexpandsize[numberofthreads - 1];
		cavebdrystartindex = oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryidx.resize(newsize);
		if (debug_error)
			printf("cavebdryexpandsize = %d\n", cavebdryexpandsize);

		kernelCavityBoundarySubsegsAppend << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_cavetetflag[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			cavebdrystartindex,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Init shrinking, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Update cavities to be star-shaped
	int cavebdrycurstartindex = 0;
	cavebdrystartindex = cavebdryexpandsize = t_cavebdrylist.size();

	numberofthreads = cavebdryexpandsize;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	t_cavebdryexpandsize.resize(numberofthreads);
	t_cavebdryexpandindices.resize(numberofthreads);
	thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);

	iteration = 0;
	while (true)
	{
		kernelUpdateCavity2StarShapedCheck << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			cavebdrycurstartindex,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());
		cavebdryexpandsize = t_cavebdryexpandindices[numberofthreads - 1] + t_cavebdryexpandsize[numberofthreads - 1];
		//if (debug_error)
		//	printf("iteration = %d, cavebdryexpandsize = %d\n", iteration, cavebdryexpandsize);
		if (cavebdryexpandsize == 0)
			break;
		iteration++;

		// Prepare memeory
		oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryidx.resize(newsize);

		kernelUpdateCavity2StarShapedAppend << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			cavebdrystartindex,
			cavebdrycurstartindex,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		numberofthreads = cavebdryexpandsize;
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		cavebdrycurstartindex = cavebdrystartindex;
		cavebdrystartindex = t_cavebdrylist.size();

		// Reset expanding lists
		t_cavebdryexpandsize.resize(numberofthreads);
		thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);
		t_cavebdryexpandindices.resize(numberofthreads);
	}

	behavior->cavebdrysizefac = t_cavebdrylist.size() * 1.0 / behavior->cavebdrysize + 0.02;
	behavior->cavebdrysize = t_cavebdrylist.size();

	if (behavior->filterstatus == 3)
	{
		behavior->filterstatus = 1;
		behavior->cavebdrysizefac = 1.1;
	}

	freeVec(t_cavetetflag);
	freeVec(t_cavebdryexpandsize);
	freeVec(t_cavebdryexpandindices);

	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	if (debug_msg >= 2)
		printf("        After cutting cavity, numberofthreads = %d, total cutting iteration = %d\n", numberofthreads, iteration);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: cavity shrinking, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Update the cavity boundary faces
	numberofthreads = t_cavebdrylist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	kernelUpdateBoundaryFaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryidx[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update the list of old tets
	numberofthreads = t_caveoldtetlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	kernelUpdateOldTets << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (t_caveshlist.size() != 0)
	{
		numberofthreads = t_caveshlist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelUpdateSubcavities << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		numberofthreads = t_threadlist.size();
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

		kernelValidateSubcavities << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_pointlocation[0]),
			thrust::raw_pointer_cast(&t_searchsh[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	// Refinement elements check
	// The new point is inserted by Delaunay refinement, i.e., it is the 
	//   circumcenter of a tetrahedron, or a subface, or a segment.
	//   Do not insert this point if the tetrahedron, or subface, or segment
	//   is not inside the final cavity.
	numberofthreads = t_threadlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	kernelValidateRefinementElements_New << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_searchsh[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_trimarker[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Reject new points if they lie too close to existing ones
	kernelCheckDistances2ClosePoints << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_searchtet[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if (debug_msg >= 1)
		printf("        After validating cavity, numberofthreads = %d(#%d, #%d, #%d)\n",
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: cavity validation, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	if (iter_0 == -1 && iter_1 == -1 && iter_2 >= behavior->miniter)
	{
		if (numberofthreads < behavior->minthread)
			return 0;
	}

	if (numberofthreads == 0)
		return 1;

	if (behavior->cavitymode == 2)
	{
		// All winners complete their cavities, reset flag if needed
		kernelResetCavityReuse << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	// Update cavity subseg list
	numberofthreads = t_cavetetseglist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateCavitySubsegs << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2tetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetseglist[0]),
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	// Update cavity subface list
	numberofthreads = t_cavetetshlist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateCavitySubfaces << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Init point insertion, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Insert points into list
	numberofthreads = t_threadlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	oldsize = t_pointradius.size();
	int oldpointsize = oldsize;
	newsize = oldsize + numberofthreads;
	t_pointlist.resize(3 * newsize);
	t_point2trilist.resize(newsize, trihandle(-1, 0));
	t_point2tetlist.resize(newsize, tethandle(-1, 11));
	t_pointtypelist.resize(newsize);
	t_pointradius.resize(newsize, 0.0);

	IntD t_threadpos(numofinsertpt, -1);
	kernelInsertNewPoints << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		thrust::raw_pointer_cast(&t_threadpos[0]),
		oldpointsize,
		numberofthreads
		);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Point insertion, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Clean up cavebdrylist first
	// remove loser from t_cavebdrylist
	numberofthreads = t_cavebdrylist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelSetCavityThreadIdx << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_cavebdryidx[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	auto first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavebdrylist.begin(), t_cavebdryidx.begin()));
	auto last_iterator =
			thrust::remove_if(first_iterator,
				thrust::make_zip_iterator(thrust::make_tuple(t_cavebdrylist.end(), t_cavebdryidx.end())),
				isInvalidCavityTuple());
	int newlistsize = thrust::distance(first_iterator, last_iterator);
	t_cavebdrylist.resize(newlistsize);
	t_cavebdryidx.resize(newlistsize);

	// Remove duplicate boundary faces in t_cavebdrylist
	first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavebdrylist.begin(), t_cavebdryidx.begin()));
	last_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavebdrylist.end(), t_cavebdryidx.end()));
	thrust::sort(first_iterator, last_iterator, CavityTupleComp());

	first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavebdrylist.begin(), t_cavebdryidx.begin()));
	last_iterator =
		thrust::unique(first_iterator,
			thrust::make_zip_iterator(thrust::make_tuple(t_cavebdrylist.end(), t_cavebdryidx.end())),
			CavityTupleEqualTo());
	newlistsize = thrust::distance(first_iterator, last_iterator);
	t_cavebdrylist.resize(newlistsize);
	t_cavebdryidx.resize(newlistsize);

	// Compute shortest edges
	RealD t_smlen(numofencsubseg + numofencsubface);
	IntD t_parentpt(numofencsubseg + numofencsubface);
	IntD t_scanleft(numofencsubseg + numofencsubface, -1), t_scanright(numofencsubseg + numofencsubface, -1);

	numberofthreads = t_cavebdrylist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelComputeShortestEdgeLength_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_cavebdryidx[0]),
		thrust::raw_pointer_cast(&t_scanleft[0]),
		thrust::raw_pointer_cast(&t_scanright[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	numberofthreads = t_threadlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelComputeShortestEdgeLength_Phase2 << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_smlen[0]),
		thrust::raw_pointer_cast(&t_parentpt[0]),
		thrust::raw_pointer_cast(&t_scanleft[0]),
		thrust::raw_pointer_cast(&t_scanright[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	freeVec(t_scanleft);
	freeVec(t_scanright);

	// Create new tetrahedra to fill the cavity
	int tetexpandsize = t_cavebdrylist.size();
	numberofthreads = tetexpandsize;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	IntD t_emptyslot;
	int numberofemptyslot = updateEmptyTets(t_tetstatus, t_emptyslot);
	if (numberofemptyslot < tetexpandsize) // dont have enough empty slots, extend lists
	{
		oldsize = t_tetstatus.size();
		newsize = oldsize + tetexpandsize - numberofemptyslot;
		try
		{
			t_tetlist.resize(4 * newsize, -1);
			t_neighborlist.resize(4 * newsize, tethandle(-1, 11));
			t_tet2trilist.resize(4 * newsize, trihandle(-1, 0));
			t_tet2seglist.resize(6 * newsize, trihandle(-1, 0));
			t_tetstatus.resize(newsize, tetstatus(0));
		}
		catch (thrust::system_error &e)
		{
			// output an error message and exit
			std::cerr << "Error: " << e.what() << std::endl;
			exit(-1);
		}
		numberofemptyslot = updateEmptyTets(t_tetstatus, t_emptyslot);
	}

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	kernelInsertNewTets << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryidx[0]),
		thrust::raw_pointer_cast(&t_point2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_emptyslot[0]),
		thrust::raw_pointer_cast(&t_threadpos[0]),
		oldpointsize,
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Connect adjacent new tetrahedra together
	kernelConnectNewTetNeighbors << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryidx[0]),
		thrust::raw_pointer_cast(&t_point2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Connect boundary subfaces to the new tets
	numberofthreads = t_cavetetshlist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelSetCavityThreadIdx << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		auto first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavetetshlist.begin(), t_cavetetshidx.begin()));
		auto last_iterator =
			thrust::remove_if(first_iterator,
				thrust::make_zip_iterator(thrust::make_tuple(t_cavetetshlist.end(), t_cavetetshidx.end())),
				isInvalidSubfaceTuple());
		newlistsize = thrust::distance(first_iterator, last_iterator);
		t_cavetetshlist.resize(newlistsize);
		t_cavetetshidx.resize(newlistsize);

		numberofthreads = t_cavetetshlist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelConnectBoundarySubfaces2NewTets << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tet2trilist[0]),
				thrust::raw_pointer_cast(&t_cavetetshlist[0]),
				thrust::raw_pointer_cast(&t_cavetetshidx[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}
	}

	// Connect boundary segments to the new tets
	numberofthreads = t_cavetetseglist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelSetCavityThreadIdx << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		auto first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavetetseglist.begin(), t_cavetetsegidx.begin()));
		auto last_iterator =
			thrust::remove_if(first_iterator,
				thrust::make_zip_iterator(thrust::make_tuple(t_cavetetseglist.end(), t_cavetetsegidx.end())),
				isInvalidSubfaceTuple());
		newlistsize = thrust::distance(first_iterator, last_iterator);
		t_cavetetseglist.resize(newlistsize);
		t_cavetetsegidx.resize(newlistsize);

		numberofthreads = t_cavetetseglist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelConnectBoundarySubsegs2NewTets << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_seg2tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tet2seglist[0]),
				thrust::raw_pointer_cast(&t_cavetetseglist[0]),
				thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
				thrust::raw_pointer_cast(&t_segmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Insert new tets, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Split a subface or a segment
	TriHandleD t_caveshbdlist;
	IntD t_caveshbdidx;
	numberofsplittablesubsegs = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0);
	numberofsplittablesubfaces = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1);
	if (numberofsplittablesubsegs != 0 || numberofsplittablesubfaces != 0)
	{
		numberofthreads = t_caveshlist.size();
		if (numberofthreads > 0)
		{
			// Clean up t_caveshlist first
			// remove loser from t_caveshlist
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelSetCavityThreadIdx << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			auto first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_caveshlist.begin(), t_caveshidx.begin()));
			auto last_iterator =
				thrust::remove_if(first_iterator,
					thrust::make_zip_iterator(thrust::make_tuple(t_caveshlist.end(), t_caveshidx.end())),
					isInvalidSubfaceTuple());
			newlistsize = thrust::distance(first_iterator, last_iterator);
			t_caveshlist.resize(newlistsize);
			t_caveshidx.resize(newlistsize);

			// Remove duplicate subfaces from t_caveshlist
			first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_caveshlist.begin(), t_caveshidx.begin()));
			last_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_caveshlist.end(), t_caveshidx.end()));
			thrust::sort(first_iterator, last_iterator, SubfaceTupleComp());

			first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_caveshlist.begin(), t_caveshidx.begin()));
			last_iterator =
				thrust::unique(first_iterator,
					thrust::make_zip_iterator(thrust::make_tuple(t_caveshlist.end(), t_caveshidx.end())),
					SubfaceTupleEqualTo());
			newlistsize = thrust::distance(first_iterator, last_iterator);
			t_caveshlist.resize(newlistsize);
			t_caveshidx.resize(newlistsize);
		}

		numberofthreads = t_caveshlist.size();
		if (numberofthreads > 0)
		{
			// Create subcavity boundary edge list
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			IntD t_caveshbdsize(numberofthreads, 0), t_caveshbdindices(numberofthreads, -1);
			kernelSubCavityBoundaryEdgeCheck << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_caveshbdsize[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			thrust::exclusive_scan(t_caveshbdsize.begin(), t_caveshbdsize.end(), t_caveshbdindices.begin());
			int caveshbdsize = t_caveshbdindices[numberofthreads - 1] + t_caveshbdsize[numberofthreads - 1];
			t_caveshbdlist.resize(caveshbdsize);
			t_caveshbdidx.resize(caveshbdsize);

			kernelSubCavityBoundaryEdgeAppend << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_caveshbdlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdidx[0]),
				thrust::raw_pointer_cast(&t_caveshbdsize[0]),
				thrust::raw_pointer_cast(&t_caveshbdindices[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				printf("caveshbdsize = %d\n", caveshbdsize);
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			// Create new subfaces
			numberofemptyslot = updateEmptyTris(t_tristatus, t_emptyslot);
			if (numberofemptyslot < caveshbdsize) // dont have enough empty slots, extend lists
			{
				oldsize = t_tristatus.size();
				newsize = oldsize + caveshbdsize - numberofemptyslot;
				t_trifacelist.resize(3 * newsize, -1);
				t_tri2tetlist.resize(2 * newsize, tethandle(-1, 11));
				t_tri2trilist.resize(3 * newsize, trihandle(-1, 0));
				t_tri2seglist.resize(3 * newsize, trihandle(-1, 0));
				t_tri2parentidxlist.resize(newsize, -1);
				t_tristatus.resize(newsize, tristatus(0));
				t_subfaceencmarker.resize(newsize, -1);
				t_trimarker.resize(newsize, 0); // extend to identify new subfaces when need to read from tristatus
				numberofemptyslot = updateEmptyTris(t_tristatus, t_emptyslot);
			}

			numberofthreads = caveshbdsize;
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			TriHandleD t_casout(numberofthreads, trihandle(-1, 0)), t_casin(numberofthreads, trihandle(-1, 0));
			kernelInsertNewSubfaces << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_point2trilist[0]),
				thrust::raw_pointer_cast(&t_pointtypelist[0]),
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_tri2parentidxlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdidx[0]),
				thrust::raw_pointer_cast(&t_emptyslot[0]),
				thrust::raw_pointer_cast(&t_casout[0]),
				thrust::raw_pointer_cast(&t_casin[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				thrust::raw_pointer_cast(&t_threadpos[0]),
				oldpointsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			kernelConnectNewSubface2OuterSubface_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_caveshbdlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdidx[0]),
				thrust::raw_pointer_cast(&t_emptyslot[0]),
				thrust::raw_pointer_cast(&t_casout[0]),
				thrust::raw_pointer_cast(&t_casin[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			kernelConnectNewSubface2OuterSubface_Phase2 << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_point2trilist[0]),
				thrust::raw_pointer_cast(&t_pointtypelist[0]),
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_caveshbdlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdidx[0]),
				thrust::raw_pointer_cast(&t_emptyslot[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				thrust::raw_pointer_cast(&t_threadpos[0]),
				oldpointsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			kernelConnectNewSubfaceNeighbors << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_caveshbdlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdidx[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		// Remove losers from t_cavesegshlist
		numberofthreads = t_cavesegshlist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelSetCavityThreadIdx << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_cavesegshidx[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			auto first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_cavesegshlist.begin(), t_cavesegshidx.begin()));
			auto last_iterator =
				thrust::remove_if(first_iterator,
					thrust::make_zip_iterator(thrust::make_tuple(t_cavesegshlist.end(), t_cavesegshidx.end())),
					isInvalidSubfaceTuple());
			newlistsize = thrust::distance(first_iterator, last_iterator);
			t_cavesegshlist.resize(newlistsize);
			t_cavesegshidx.resize(newlistsize);
		}

		// Remove degenerated subfaces at segments
		numberofthreads = t_cavesegshlist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelRemoveDegeneratedNewSubfaces << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_point2trilist[0]),
				thrust::raw_pointer_cast(&t_pointtypelist[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_cavesegshlist[0]),
				thrust::raw_pointer_cast(&t_cavesegshidx[0]),
				thrust::raw_pointer_cast(&t_initialsubcavitysize[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				thrust::raw_pointer_cast(&t_threadpos[0]),
				oldpointsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		if (numberofsplittablesubsegs != 0)
		{
			// Create new subsegs
			numberofemptyslot = updateEmptyTris(t_segstatus, t_emptyslot);
			if (numberofemptyslot < 2 * numberofsplittablesubsegs) // dont have enough empty slots, extend lists
			{
				oldsize = t_segstatus.size();
				newsize = oldsize + 2 * numberofsplittablesubsegs - numberofemptyslot;
				t_seglist.resize(3 * newsize, -1);
				t_seg2trilist.resize(3 * newsize, trihandle(-1, 0));
				t_seg2tetlist.resize(newsize, tethandle(-1, 11));
				t_seg2parentidxlist.resize(newsize, -1);
				t_segstatus.resize(newsize, tristatus(0));
				t_segmarker.resize(newsize, 0); // extend to identify new subsegs when need to read from segstatus
				t_segencmarker.resize(newsize, -1);
				numberofemptyslot = updateEmptyTris(t_segstatus, t_emptyslot);
			}

			numberofthreads = t_threadlist.size();
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelInsertNewSubsegs << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_point2trilist[0]),
				thrust::raw_pointer_cast(&t_pointtypelist[0]),
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_seg2parentidxlist[0]),
				thrust::raw_pointer_cast(&t_segstatus[0]),
				thrust::raw_pointer_cast(&t_segencmarker[0]),
				thrust::raw_pointer_cast(&t_emptyslot[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				oldpointsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}

			numberofthreads = t_cavesegshlist.size();
			if (numberofthreads > 0)
			{
				numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
				kernelConnectNewSubseg2NewSubface << < numberofblocks, BLOCK_SIZE >> > (
					thrust::raw_pointer_cast(&t_insertidxlist[0]),
					thrust::raw_pointer_cast(&t_threadlist[0]),
					thrust::raw_pointer_cast(&t_seglist[0]),
					thrust::raw_pointer_cast(&t_seg2trilist[0]),
					thrust::raw_pointer_cast(&t_segstatus[0]),
					thrust::raw_pointer_cast(&t_trifacelist[0]),
					thrust::raw_pointer_cast(&t_tri2trilist[0]),
					thrust::raw_pointer_cast(&t_tri2seglist[0]),
					thrust::raw_pointer_cast(&t_cavesegshlist[0]),
					thrust::raw_pointer_cast(&t_cavesegshidx[0]),
					thrust::raw_pointer_cast(&t_emptyslot[0]),
					thrust::raw_pointer_cast(&t_threadmarker[0]),
					thrust::raw_pointer_cast(&t_threadpos[0]),
					oldpointsize,
					numberofthreads
					);

				if (debug_error)
				{
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
				}
			}

			// Connect new subsegs to outer subsegs and collect new subsegs into list
			t_cavesegshlist.resize(2 * numberofsplittablesubsegs);
			t_cavesegshidx.resize(2 * numberofsplittablesubsegs);
			numberofthreads = t_threadlist.size();
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelConnectNewSubseg2OuterSubseg << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_segmarker[0]),
				thrust::raw_pointer_cast(&t_cavesegshlist[0]),
				thrust::raw_pointer_cast(&t_cavesegshidx[0]),
				thrust::raw_pointer_cast(&t_emptyslot[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		// Recover new subfaces in cavities
		numberofthreads = t_caveshbdlist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelConnectNewSubfaces2NewTets << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tet2trilist[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_caveshbdlist[0]),
				thrust::raw_pointer_cast(&t_caveshbdidx[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				thrust::raw_pointer_cast(&t_threadpos[0]),
				oldpointsize,
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}

		if (numberofsplittablesubsegs != 0)
		{
			numberofthreads = t_cavesegshlist.size();
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelConnectNewSubsegs2NewTets << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_point2tetlist[0]),
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_seg2tetlist[0]),
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tet2seglist[0]),
				thrust::raw_pointer_cast(&t_cavesegshlist[0]),
				thrust::raw_pointer_cast(&t_cavesegshidx[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Insert new subsegments and subfaces, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Update encroachment markers
	numberofthreads = t_cavetetseglist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateSegencmarker_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2tetlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavetetseglist[0]),
			thrust::raw_pointer_cast(&t_cavetetsegidx[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_segencmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	numberofthreads = t_cavesegshlist.size();
	if (numberofthreads > 0 && numberofsplittablesubsegs != 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateSegencmarker_Phase2 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2tetlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavesegshlist[0]),
			thrust::raw_pointer_cast(&t_cavesegshidx[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_segencmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	numberofthreads = t_cavetetshlist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateSubfaceencmarker_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshlist[0]),
			thrust::raw_pointer_cast(&t_cavetetshidx[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	numberofthreads = t_caveshbdlist.size();
	if (numberofthreads > 0)
	{
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelUpdateSubfaceencmarker_Phase2 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdidx[0]),
			thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	// Update tet bad status
	numberofthreads = t_cavebdrylist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelUpdateTetBadstatus << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryidx[0]),
		behavior->radius_to_edge_ratio,
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update insertion radius after insertion
	numberofthreads = t_threadlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelUpdateInsertRadius << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_point2trilist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_pointradius[0]),
		thrust::raw_pointer_cast(&t_seg2parentidxlist[0]),
		thrust::raw_pointer_cast(&t_segparentendpointidxlist[0]),
		thrust::raw_pointer_cast(&t_tri2parentidxlist[0]),
		thrust::raw_pointer_cast(&t_triid2parentoffsetlist[0]),
		thrust::raw_pointer_cast(&t_triparentendpointidxlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_smlen[0]),
		thrust::raw_pointer_cast(&t_parentpt[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		oldpointsize,
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Clear old elements' information
	if (numberofsplittablesubsegs != 0)
	{
		kernelResetOldSubsegInfo << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	if (numberofsplittablesubsegs != 0 || numberofsplittablesubfaces != 0)
	{
		numberofthreads = t_caveshlist.size();
		if (numberofthreads > 0)
		{
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			kernelResetOldSubfaceInfo << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshidx[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			if (debug_error)
			{
				gpuErrchk(cudaPeekAtLastError());
				gpuErrchk(cudaDeviceSynchronize());
			}
		}
	}

	// Clean up t_caveoldtetlist
	// remove loser from t_caveoldtetlist
	numberofthreads = t_caveoldtetlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelSetCavityThreadIdx << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	first_iterator = thrust::make_zip_iterator(thrust::make_tuple(t_caveoldtetlist.begin(), t_caveoldtetidx.begin()));
	last_iterator =
		thrust::remove_if(first_iterator,
			thrust::make_zip_iterator(thrust::make_tuple(t_caveoldtetlist.end(), t_caveoldtetidx.end())),
			isInvalidCavityTuple());
	newlistsize = thrust::distance(first_iterator, last_iterator);
	t_caveoldtetlist.resize(newlistsize);
	t_caveoldtetidx.resize(newlistsize);

	numberofthreads = t_caveoldtetlist.size();
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelResetOldTetInfo << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetidx[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update the numbers of mesh elements
	numofpoints = t_pointradius.size();
	numofsubseg = t_segstatus.size();
	numofsubface = t_tristatus.size();
	numoftet = t_tetstatus.size();

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Update status, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Check neighbors
	if (debug_error)
	{
		numberofthreads = t_pointtypelist.size();
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelCheckPointNeighbors << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_point2trilist[0]),
			thrust::raw_pointer_cast(&t_point2tetlist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			numberofthreads
			);

		numberofthreads = t_segstatus.size();
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelCheckSubsegNeighbors << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			thrust::raw_pointer_cast(&t_seg2tetlist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_tet2seglist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			numberofthreads
			);

		numberofthreads = t_tristatus.size();
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelCheckSubfaceNeighbors << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_tet2trilist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			numberofthreads
			);

		numberofthreads = t_tetstatus.size();
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		kernelCheckTetNeighbors << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2tetlist[0]),
			thrust::raw_pointer_cast(&t_segstatus[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tet2trilist[0]),
			thrust::raw_pointer_cast(&t_tet2seglist[0]),
			thrust::raw_pointer_cast(&t_tetstatus[0]),
			numberofthreads
			);

		//int numofunsplittabletets = thrust::count_if(t_tetstatus.begin(), t_tetstatus.end(), isAbortiveTet());
		//printf("number of unsplittable tets = %d\n", numofunsplittabletets);
	}

	return 1;
}