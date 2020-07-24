#include "CudaInsertPoint.h"
#include "CudaSplitEncseg.h"
#include "CudaMesh.h"
#include "CudaAnimation.h"
#include <math_constants.h>
#include <time.h>

int insertPoint(
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
	IntD& t_threadmarker,
	int numofinsertpt,
	int numofencsubseg,
	int numofencsubface,
	int numofbadtet,
	int& numofpoints,
	int& numofsubseg,
	int& numofsubface,
	int& numoftet,
	MESHBH* behavior,
	int iter_0,
	int iter_1,
	int iter_2,
	int debug_msg,
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

	// Initialization
	int numberofthreads;
	int numberofblocks;
	int numberofsplittablesubsegs;
	int numberofsplittablesubfaces;
	int numberofsplittabletets;
	IntD t_threadlist; // active thread list
	UInt64D t_tetmarker(numoftet, MAXULL); // marker for tets. Used for cavity.
	IntD t_segmarker(numofsubseg, MAXINT); // marker for subsegs. Used for splitting segments
	UInt64D t_trimarker(numofsubface, MAXULL); // marker for subfaces. Used for subcavity.

	RealD t_insertptlist(3*numofinsertpt);
	IntD t_priority(numofinsertpt, 0);
	RealD t_tmpreal(numofinsertpt, 0.0); // store real temporarily
	thrust::device_vector<locateresult> t_pointlocation(numofinsertpt, UNKNOWN);
	ULongD t_randomseed;
	TetHandleD t_searchtet;
	TriHandleD t_searchsh;

	// Compute Steiner points and priorities
	numberofblocks = (ceil)((float)numofinsertpt / BLOCK_SIZE);
	kernelComputeSteinerPointAndPriority << <numberofblocks, BLOCK_SIZE >> > (
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
		thrust::raw_pointer_cast(&t_tmpreal[0]),
		numofinsertpt
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Modify priorities and convert them into integers
	// subsegment > subface > tet
	// Normalize priority for subsegment to [1,2]
	// subface to [3,4] and tet to [5,6]
	double priority_min[3], priority_max[3], priority_offset[3] = {0, 0, 0};
	thrust::pair<RealD::iterator, RealD::iterator> priority_pair;
	if (numofencsubseg > 0)
	{
		priority_pair = 
			thrust::minmax_element(
				t_tmpreal.begin(), 
				t_tmpreal.begin() + numofencsubseg);
		priority_min[0] = *priority_pair.first;
		priority_max[0] = *priority_pair.second;
		priority_offset[0] = 0;
		if (debug_error)
		{
			printf("MinMax Real priorities for subseg: %f, %f\n", priority_min[0], priority_max[0]);
			printf("Offset: %f\n", priority_offset[0]);
		}
	}
	if (numofencsubface > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_tmpreal.begin() + numofencsubseg, 
				t_tmpreal.begin() + numofencsubseg + numofencsubface);
		priority_min[1] = *priority_pair.first;
		priority_max[1] = *priority_pair.second;
		if (numofencsubseg > 0)
			priority_offset[1] = priority_max[0] + 1 - priority_min[1];
		else
			priority_offset[1] = 0;
		if (debug_error)
		{
			printf("MinMax Real priorities for subface: %f, %f\n", priority_min[1], priority_max[1]);
			printf("Offset: %f\n", priority_offset[1]);
		}
	}
	if (numofbadtet > 0)
	{
		priority_pair =
			thrust::minmax_element(
				t_tmpreal.begin() + numofencsubseg + numofencsubface,
				t_tmpreal.end());
		priority_min[2] = *priority_pair.first;
		priority_max[2] = *priority_pair.second;
		if (numofencsubface > 0)
			priority_offset[2] = priority_max[1] + priority_offset[1] + 1 - priority_min[2];
		else if (numofencsubseg > 0)
			priority_offset[2] = priority_max[0] + 1 - priority_min[2];
		else
			priority_offset[2] = 0;
		if (debug_error)
		{
			printf("MinMax Real priorities for tet: %f, %f\n", priority_min[2], priority_max[2]);
			printf("Offset: %f\n", priority_offset[2]);
		}
	}
	kernelModifyPriority << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_tmpreal[0]),
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

	// Reject points when violate insertradius rules
	kernelCheckInsertRadius<< <numberofblocks, BLOCK_SIZE >> >(
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

	//printf("%d, ", numberofthreads);

	// Locate Steiner points
	numberofsplittablesubsegs = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0);
	t_searchtet.resize(numofinsertpt, tethandle(-1, 11));
	t_searchsh.resize(numofencsubseg + numofencsubface, trihandle(-1, 11)); // only used in subfaces
	t_randomseed.resize(numberofthreads);
	thrust::fill(t_randomseed.begin(), t_randomseed.end(), 1);
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
		thrust::raw_pointer_cast(&t_randomseed[0]),
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
		if(iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
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
		printf("        Stage: Initialization, time = %f\n", (REAL)(tv[1]-tv[0]));
	}

	IntD t_linklistcur(numofinsertpt, -1);
	IntD t_cavethreadidx;
	IntD t_threadfinishmarker(numofinsertpt);

	TetHandleD t_caveoldtetlist; // linklist to record interior tets
	IntD t_caveoldtetprev, t_caveoldtetnext;
	IntD t_caveoldtethead(numofinsertpt, -1), t_caveoldtettail(numofinsertpt, -1);
	TetHandleD t_cavetetlist; // linklist to record tets in expanding cavities
	IntD t_cavetetprev, t_cavetetnext;
	IntD t_cavetethead(numofinsertpt, -1), t_cavetettail(numofinsertpt, -1);
	TriHandleD t_caveshlist; // linklist for subcavities
	IntD t_caveshprev, t_caveshnext;
	IntD t_caveshhead(numofinsertpt, -1), t_caveshtail(numofinsertpt, -1);
	TriHandleD t_cavesegshlist; // linklist for face-at-splitedges
	IntD t_cavesegshprev, t_cavesegshnext;
	IntD t_cavesegshhead(numofinsertpt, -1), t_cavesegshtail(numofinsertpt, -1);

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
	initialcavitysize = thrust::reduce(t_initialcavitysize.begin(), t_initialcavitysize.end());
	initialsubcavitysize = thrust::reduce(t_initialsubcavitysize.begin(), t_initialsubcavitysize.end());
	thrust::exclusive_scan(t_initialcavitysize.begin(), t_initialcavitysize.end(), t_initialcavityindices.begin());
	thrust::exclusive_scan(t_initialsubcavitysize.begin(), t_initialsubcavitysize.end(), t_initialsubcavityindices.begin());

	// init cavity linklists
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	t_caveoldtetlist.resize(initialcavitysize);
	t_caveoldtetprev.resize(initialcavitysize);
	t_caveoldtetnext.resize(initialcavitysize);
	int expandfactor = 4;
	t_cavetetlist.resize(expandfactor * initialcavitysize);
	t_cavetetprev.resize(expandfactor * initialcavitysize);
	t_cavetetnext.resize(expandfactor * initialcavitysize);
	t_cavethreadidx.resize(expandfactor * initialcavitysize);
	thrust::fill(t_cavethreadidx.begin(), t_cavethreadidx.end(), -1);

	if (initialsubcavitysize > 0)
	{
		t_caveshlist.resize(initialsubcavitysize);
		t_caveshprev.resize(initialsubcavitysize);
		t_caveshnext.resize(initialsubcavitysize);
		t_cavesegshlist.resize(initialsubcavitysize);
		t_cavesegshprev.resize(initialsubcavitysize);
		t_cavesegshnext.resize(initialsubcavitysize);
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
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_caveoldtettail[0]),
		thrust::raw_pointer_cast(&t_cavetetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetprev[0]),
		thrust::raw_pointer_cast(&t_cavetetnext[0]),
		thrust::raw_pointer_cast(&t_cavetethead[0]),
		thrust::raw_pointer_cast(&t_cavetettail[0]),
		thrust::raw_pointer_cast(&t_initialsubcavityindices[0]),
		thrust::raw_pointer_cast(&t_initialsubcavitysize[0]),
		thrust::raw_pointer_cast(&t_cavethreadidx[0]),
		thrust::raw_pointer_cast(&t_caveshlist[0]),
		thrust::raw_pointer_cast(&t_caveshprev[0]),
		thrust::raw_pointer_cast(&t_caveshnext[0]),
		thrust::raw_pointer_cast(&t_caveshhead[0]),
		thrust::raw_pointer_cast(&t_caveshtail[0]),
		thrust::raw_pointer_cast(&t_cavesegshlist[0]),
		thrust::raw_pointer_cast(&t_cavesegshprev[0]),
		thrust::raw_pointer_cast(&t_cavesegshnext[0]),
		thrust::raw_pointer_cast(&t_cavesegshhead[0]),
		thrust::raw_pointer_cast(&t_cavesegshtail[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (drawmesh != NULL && drawmesh->animation)
	{
		// Output animation frame mesh
		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
			iter_2 == drawmesh->iter_tet)
		{
			outputCavityFrame(
				drawmesh,
				t_pointlist,
				t_tetlist,
				t_tetmarker,
				t_threadmarker,
				t_caveoldtetlist,
				t_caveoldtetnext,
				t_caveoldtethead,
				iter_0,
				iter_1,
				iter_2,
				-1,
				initialcavitysize
			);
		}
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
	TetHandleD t_cavebdrylist; // linklist to record interior tets
	IntD t_cavebdryprev, t_cavebdrynext;
	IntD t_cavebdryhead(numofinsertpt, -1), t_cavebdrytail(numofinsertpt, -1);

	int cavetetcurstartindex = 0;
	int cavetetstartindex = t_cavetetlist.size();
	int caveoldtetstartindex = t_caveoldtetlist.size();
	int cavebdrystartindex = 0;
	int cavetetexpandsize = cavetetstartindex, caveoldtetexpandsize, cavebdryexpandsize;

	numberofthreads = cavetetexpandsize; // each thread works on one tet in cavetetlist
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	IntD t_cavetetexpandsize(numberofthreads, 0);
	IntD t_caveoldtetexpandsize(numberofthreads, 0);
	IntD t_cavebdryexpandsize(numberofthreads, 0);
	IntD t_cavetetexpandindices(numberofthreads, -1);
	IntD t_caveoldtetexpandindices(numberofthreads, -1);
	IntD t_cavebdryexpandindices(numberofthreads, -1);
	IntD t_cavetetthreadidx;
	IntD t_caveoldtetthreadidx;
	IntD t_cavebdrythreadidx;

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
	}
	int iteration = 0;
	while (true)
	{
		if (iteration > behavior->maxcavity) // Too large cavities. Stop and mark as abortive elements
		{
			kernelLargeCavityCheck << < numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_insertptlist[0]),
				thrust::raw_pointer_cast(&t_cavethreadidx[0]),
				thrust::raw_pointer_cast(&t_segstatus[0]),
				thrust::raw_pointer_cast(&t_tristatus[0]),
				thrust::raw_pointer_cast(&t_tetstatus[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);

			break;
		}

		// Check if current tet is included in cavity
		kernelCavityExpandingCheck << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetprev[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetethead[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
			thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
			thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
			thrust::raw_pointer_cast(&t_caveoldtethead[0]),
			thrust::raw_pointer_cast(&t_caveoldtettail[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryprev[0]),
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdryhead[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
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
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		// Count expanding sizes and indices
		// these reduce functons are possible to be removed by the last elements of prefix sums(optimazation)
		cavetetexpandsize = thrust::reduce(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end());
		caveoldtetexpandsize = thrust::reduce(t_caveoldtetexpandsize.begin(), t_caveoldtetexpandsize.end());
		cavebdryexpandsize = thrust::reduce(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end());
		thrust::exclusive_scan(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), t_cavetetexpandindices.begin());
		thrust::exclusive_scan(t_caveoldtetexpandsize.begin(), t_caveoldtetexpandsize.end(), t_caveoldtetexpandindices.begin());
		thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());

		// Prepare memeory
		oldsize = t_cavetetlist.size();
		newsize = oldsize + cavetetexpandsize;
		t_cavetetlist.resize(newsize);
		t_cavetetprev.resize(newsize);
		t_cavetetnext.resize(newsize);
		t_cavetetthreadidx.resize(cavetetexpandsize);
		oldsize = t_caveoldtetlist.size();
		newsize = oldsize + caveoldtetexpandsize;
		t_caveoldtetlist.resize(newsize);
		t_caveoldtetprev.resize(newsize);
		t_caveoldtetnext.resize(newsize);
		t_caveoldtetthreadidx.resize(caveoldtetexpandsize);
		oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryprev.resize(newsize);
		t_cavebdrynext.resize(newsize);
		t_cavebdrythreadidx.resize(cavebdryexpandsize);

		// Set threadidx list
		kernelCavityExpandingSetThreadidx << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavetetthreadidx[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			thrust::raw_pointer_cast(&t_caveoldtetthreadidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavebdrythreadidx[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		// Mark and append expanding tets
		kernelCavityExpandingMarkAndAppend << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetprev[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetethead[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavetetthreadidx[0]),
			cavetetstartindex,
			cavetetexpandsize,
			thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
			thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
			thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
			thrust::raw_pointer_cast(&t_caveoldtethead[0]),
			thrust::raw_pointer_cast(&t_caveoldtettail[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			thrust::raw_pointer_cast(&t_caveoldtetthreadidx[0]),
			caveoldtetstartindex,
			caveoldtetexpandsize,
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryprev[0]),
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdryhead[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavebdrythreadidx[0]),
			cavebdrystartindex,
			cavebdryexpandsize,
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

		kernelCavityExpandingUpdateListTails << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			cavetetstartindex,
			thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
			thrust::raw_pointer_cast(&t_caveoldtettail[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandsize[0]),
			thrust::raw_pointer_cast(&t_caveoldtetexpandindices[0]),
			caveoldtetstartindex,
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			cavebdrystartindex,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		if (drawmesh != NULL && drawmesh->animation)
		{
			// Output animation frame mesh
			if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
				iter_2 == drawmesh->iter_tet)
			{
				outputCavityFrame(
					drawmesh,
					t_pointlist,
					t_tetlist,
					t_tetmarker,
					t_threadmarker,
					t_caveoldtetlist,
					t_caveoldtetnext,
					t_caveoldtethead,
					iter_0,
					iter_1,
					iter_2,
					iteration,
					caveoldtetexpandsize
				);
			}
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

		// Set threadidx list
		t_cavethreadidx.resize(numberofthreads);
		thrust::copy(t_cavetetthreadidx.begin(), t_cavetetthreadidx.end(), t_cavethreadidx.begin());

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

		//if(debug_timing)
		//{
		//	cudaDeviceSynchronize();
		//	printf("time = %lf\n", (REAL)(clock() - tv[0]));
		//}
	}

	// Update working threadlist to winners
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if(debug_msg >= 1)
		printf("        After expanding cavity, numberofthreads = %d(#%d, #%d, #%d), total expanding iteration = %d\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2),
			iteration);
	if (numberofthreads == 0)
	{
		//printf("Error: 0 threads after expanding cavities!\n");
		//exit(0);
		return 0;
	}

	if (drawmesh != NULL && drawmesh->animation)
	{
		// Output animation frame mesh
		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
			iter_2 == drawmesh->iter_tet)
		{
			outputCavityFrame(
				drawmesh,
				t_pointlist,
				t_trifacelist,
				t_tristatus,
				t_tri2tetlist,
				t_tetlist,
				t_neighborlist,
				t_tetmarker,
				t_threadmarker,
				t_caveoldtetlist,
				t_caveoldtetnext,
				t_caveoldtethead,
				iter_0,
				iter_1,
				iter_2,
				-1,
				0
			);
		}
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Cavity expanding, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Collect segments in cavities
	TriHandleD t_cavetetseglist;
	IntD t_cavetetsegprev, t_cavetetsegnext;
	IntD t_cavetetseghead(numofinsertpt, -1), t_cavetetsegtail(numofinsertpt, -1);
	IntD t_cavetetsegsize(numberofthreads, 0), t_cavetetsegindices(numberofthreads, -1);
	IntD t_segmarker2(numofsubseg, MAXINT);

	kernelMarkCavityAdjacentSubsegs << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads,
		thrust::raw_pointer_cast(&t_tetmarker[0])
		);

	kernelCountCavitySubsegs << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_cavetetsegsize[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	thrust::exclusive_scan(t_cavetetsegsize.begin(), t_cavetetsegsize.end(), t_cavetetsegindices.begin());
	int cavetetsegsize = thrust::reduce(t_cavetetsegsize.begin(), t_cavetetsegsize.end());
	t_cavetetseglist.resize(cavetetsegsize);
	t_cavetetsegprev.resize(cavetetsegsize);
	t_cavetetsegnext.resize(cavetetsegsize);
	kernelAppendCavitySubsegs << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegprev[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_cavetetsegtail[0]),
		thrust::raw_pointer_cast(&t_cavetetsegsize[0]),
		thrust::raw_pointer_cast(&t_cavetetsegindices[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		numberofthreads
		);

	// remove the cavities that share segments
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	// Segment encroachment test
	kernelCheckSegmentEncroachment << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_segencmarker[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

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

	// Collect subfaces in cavities
	TriHandleD t_cavetetshlist;
	IntD t_cavetetshprev, t_cavetetshnext;
	IntD t_cavetetshhead(numofinsertpt, -1), t_cavetetshtail(numofinsertpt, -1);
	IntD t_cavetetshsize(numberofthreads, 0), t_cavetetshindices(numberofthreads, -1);
	IntD t_trimarker2(numofsubface, MAXINT);

	kernelMarkCavityAdjacentSubfaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_trimarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	kernelCountCavitySubfaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_cavetetshsize[0]),
		thrust::raw_pointer_cast(&t_trimarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	thrust::exclusive_scan(t_cavetetshsize.begin(), t_cavetetshsize.end(), t_cavetetshindices.begin());
	int cavetetshsize = thrust::reduce(t_cavetetshsize.begin(), t_cavetetshsize.end());
	t_cavetetshlist.resize(cavetetshsize);
	t_cavetetshprev.resize(cavetetshsize);
	t_cavetetshnext.resize(cavetetshsize);
	kernelAppendCavitySubfaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshprev[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_cavetetshtail[0]),
		thrust::raw_pointer_cast(&t_cavetetshsize[0]),
		thrust::raw_pointer_cast(&t_cavetetshindices[0]),
		thrust::raw_pointer_cast(&t_trimarker2[0]),
		numberofthreads
		);

	// remove the cavities that share subfaces
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	// Subface encroachment test
	kernelCheckSubfaceEncroachment << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_pointlocation[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

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

	if (drawmesh != NULL && drawmesh->animation)
	{
		// Output animation frame mesh
		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
			iter_2 == drawmesh->iter_tet)
		{
			outputCavityFrame(
				drawmesh,
				t_pointlist,
				t_trifacelist,
				t_tristatus,
				t_tri2tetlist,
				t_tetlist,
				t_neighborlist,
				t_tetmarker,
				t_threadmarker,
				t_caveoldtetlist,
				t_caveoldtetnext,
				t_caveoldtethead,
				iter_0,
				iter_1,
				iter_2,
				-1,
				0
			);

			outputCavityFrame(
				drawmesh,
				t_pointlist,
				t_tetlist,
				t_tetmarker,
				t_threadmarker,
				t_caveoldtetlist,
				t_caveoldtetnext,
				t_caveoldtethead,
				iter_0,
				iter_1,
				iter_2,
				-1,
				0
			);
		}
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Subsegment and subface collection, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Expand subcavities
	if (t_caveshlist.size() != 0)
	{
		thrust::copy(t_threadmarker.begin(), t_threadmarker.end(), t_threadfinishmarker.begin());

		// Init linklist current pointer
		kernelInitLinklistCurPointer << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_caveshhead[0]),
			thrust::raw_pointer_cast(&t_linklistcur[0]),
			numberofthreads
			);

		// Allocate memory
		IntD t_caveshexpandsize(numberofthreads, 0);
		IntD t_caveshexpandflag(numberofthreads, -1);
		IntD t_caveshexpandindices(numberofthreads, -1);

		// Expanding loop
		iteration = 0;
		int caveshexpandsize;
		int caveshstartindex = t_caveshlist.size();
		while (true)
		{
			//printf("iteration = %d, number of threads = %d\n", iteration, numberofthreads);
			// Check if current subface is included in subcavity
			kernelSubCavityExpandingCheck << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_insertptlist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_linklistcur[0]),
				thrust::raw_pointer_cast(&t_caveshexpandsize[0]),
				thrust::raw_pointer_cast(&t_caveshexpandflag[0]),
				thrust::raw_pointer_cast(&t_priority[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
				numberofthreads
				);

			// Count expanding sizes and indices
			caveshexpandsize = thrust::reduce(t_caveshexpandsize.begin(), t_caveshexpandsize.end());
			thrust::exclusive_scan(t_caveshexpandsize.begin(), t_caveshexpandsize.end(), t_caveshexpandindices.begin());
			//if(debug_error)
			//	printf("iteration = %d, caveshexpandsize = %d\n", iteration, caveshexpandsize);

			// Prepare memeory
			oldsize = t_caveshlist.size();
			newsize = oldsize + caveshexpandsize;
			t_caveshlist.resize(newsize);
			t_caveshprev.resize(newsize);
			t_caveshnext.resize(newsize);

			// Append expanding subfaces
			kernelSubCavityExpandingAppend << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_caveshlist[0]),
				thrust::raw_pointer_cast(&t_caveshprev[0]),
				thrust::raw_pointer_cast(&t_caveshnext[0]),
				thrust::raw_pointer_cast(&t_caveshhead[0]),
				thrust::raw_pointer_cast(&t_caveshtail[0]),
				thrust::raw_pointer_cast(&t_linklistcur[0]),
				thrust::raw_pointer_cast(&t_caveshexpandsize[0]),
				thrust::raw_pointer_cast(&t_caveshexpandindices[0]),
				thrust::raw_pointer_cast(&t_caveshexpandflag[0]),
				caveshstartindex,
				thrust::raw_pointer_cast(&t_threadfinishmarker[0]),
				numberofthreads
				);

			// Update working thread list
			numberofthreads = updateActiveListByMarker_Slot(t_threadfinishmarker, t_threadlist, t_threadfinishmarker.size());
			iteration++;
			if (numberofthreads == 0)
				break;

			// Update variables
			numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
			caveshstartindex = t_caveshlist.size();

			// Reset expanding lists
			t_caveshexpandsize.resize(numberofthreads);
			thrust::fill(t_caveshexpandsize.begin(), t_caveshexpandsize.end(), 0);
			t_caveshexpandindices.resize(numberofthreads);

			//cudaDeviceSynchronize();
		}
		if(debug_msg >= 2)
			printf("        Subcavity total expanding iteration = %d\n", iteration);
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Subcavity expanding, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Find boundary subfaces
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	IntD t_cutcount(numofinsertpt, 0);

	t_cavebdryexpandsize.resize(numberofthreads);
	t_cavebdryexpandindices.resize(numberofthreads);
	IntD t_cavetetshmarker(t_cavetetshlist.size(), -1); // marker for cavetetsh, used to mark 
	TetHandleD t_cavetetshflag(t_cavetetshlist.size(), tethandle(-1, 11));
	kernelCavityBoundarySubfacesCheck << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_cavetetshmarker[0]),
		thrust::raw_pointer_cast(&t_cavetetshflag[0]),
		thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
		thrust::raw_pointer_cast(&t_cutcount[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_trimarker[0]),
		numberofthreads
		);
	thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());
	cavebdryexpandsize = thrust::reduce(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end());
	cavebdrystartindex = oldsize = t_cavebdrylist.size();
	newsize = oldsize + cavebdryexpandsize;
	t_cavebdrylist.resize(newsize);
	t_cavebdryprev.resize(newsize);
	t_cavebdrynext.resize(newsize);
	kernelCavityBoundarySubfacesAppend << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_cavetetshmarker[0]),
		thrust::raw_pointer_cast(&t_cavetetshflag[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryprev[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_cavebdrytail[0]),
		thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
		thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
		cavebdrystartindex,
		numberofthreads
	);

	// Find boundary segments
	IntD t_cavetetsegmarker(t_cavetetseglist.size(), -1);
	TetHandleD t_cavetetsegflag(t_cavetetseglist.size(), tethandle(-1, 11));
	kernelCavityBoundarySubsegsCheck << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_cavetetsegmarker[0]),
		thrust::raw_pointer_cast(&t_cavetetsegflag[0]),
		thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
		thrust::raw_pointer_cast(&t_cutcount[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_segmarker[0]),
		numberofthreads
		);
	thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());
	cavebdryexpandsize = thrust::reduce(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end());
	cavebdrystartindex = oldsize = t_cavebdrylist.size();
	newsize = oldsize + cavebdryexpandsize;
	t_cavebdrylist.resize(newsize);
	t_cavebdryprev.resize(newsize);
	t_cavebdrynext.resize(newsize);
	kernelCavityBoundarySubsegsAppend << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_cavetetsegmarker[0]),
		thrust::raw_pointer_cast(&t_cavetetsegflag[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryprev[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_cavebdrytail[0]),
		thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
		thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
		cavebdrystartindex,
		numberofthreads
		);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Init shrinking, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Update cavities to be star-shaped

	// Sort out boundary list first so that all elements from the same cavities are put together.
	// Use cavetetlist temporally

	t_cavetetexpandsize.resize(numberofthreads); // Used for counting boundary list first
	t_cavetetexpandindices.resize(numberofthreads);
	kernelUpdateCavity2StarShapedSortOutBoundaryListCount << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
		numberofthreads
		);
	cavetetexpandsize = thrust::reduce(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end());
	thrust::exclusive_scan(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), t_cavetetexpandindices.begin());

	t_cavethreadidx.resize(cavetetexpandsize);
	t_cavetetlist.resize(cavetetexpandsize);
	t_cavetetprev.resize(cavetetexpandsize);
	t_cavetetnext.resize(cavetetexpandsize);
	kernelUpdateCavity2StarShapedSortOutBoundaryListAppend << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryprev[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_cavebdrytail[0]),
		thrust::raw_pointer_cast(&t_cavetetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetprev[0]),
		thrust::raw_pointer_cast(&t_cavetetnext[0]),
		thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
		thrust::raw_pointer_cast(&t_cavethreadidx[0]),
		numberofthreads
		);

	t_cavebdrylist.resize(cavetetexpandsize);
	t_cavebdryprev.resize(cavetetexpandsize);
	t_cavebdrynext.resize(cavetetexpandsize);
	thrust::copy(t_cavetetlist.begin(), t_cavetetlist.end(), t_cavebdrylist.begin());
	thrust::copy(t_cavetetprev.begin(), t_cavetetprev.end(), t_cavebdryprev.begin());
	thrust::copy(t_cavetetnext.begin(), t_cavetetnext.end(), t_cavebdrynext.begin());

	// Update variables and arrays
	cavetetstartindex = 0;
	int cavebdrycurstartindex = 0;
	cavebdrystartindex = cavebdryexpandsize = t_cavebdrylist.size();

	numberofthreads = cavebdryexpandsize;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	t_cavetetexpandsize.resize(numberofthreads);
	t_cavetetexpandindices.resize(numberofthreads);
	t_cavebdryexpandsize.resize(numberofthreads);
	t_cavebdryexpandindices.resize(numberofthreads);

	t_cavetetlist.resize(0); // clear lists
	t_cavetetprev.resize(0);
	t_cavetetnext.resize(0);
	thrust::fill(t_cavetethead.begin(), t_cavetethead.end(), -1);
	thrust::fill(t_cavetettail.begin(), t_cavetettail.end(), -1);

	iteration = 0;
	while (true)
	{
		//printf("iteration = %d, numberofthreads = %d\n", iteration, numberofthreads);

		// Count valid exterior tets and tets to be removed
		kernelUpdateCavity2StarShapedCheck << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_insertidxlist[0]),
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_tetlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tet2seglist[0]),
			thrust::raw_pointer_cast(&t_insertptlist[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryprev[0]),
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdryhead[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cutcount[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			cavebdrycurstartindex,
			numberofthreads
			);

		// Count expanding sizes and indices
		cavetetexpandsize = thrust::reduce(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end());
		cavebdryexpandsize = thrust::reduce(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end());
		thrust::exclusive_scan(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), t_cavetetexpandindices.begin());
		thrust::exclusive_scan(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), t_cavebdryexpandindices.begin());
		//if(debug_error)
		//	printf("iteration = %d, cavebdryexpandsize = %d\n", iteration, cavebdryexpandsize);

		// Prepare memeory
		oldsize = t_cavetetlist.size();
		newsize = oldsize + cavetetexpandsize;
		t_cavetetlist.resize(newsize);
		t_cavetetprev.resize(newsize);
		t_cavetetnext.resize(newsize);
		t_cavetetthreadidx.resize(cavetetexpandsize);
		oldsize = t_cavebdrylist.size();
		newsize = oldsize + cavebdryexpandsize;
		t_cavebdrylist.resize(newsize);
		t_cavebdryprev.resize(newsize);
		t_cavebdrynext.resize(newsize);
		t_cavebdrythreadidx.resize(cavebdryexpandsize);

		kernelUpdateCavity2StarShapedSetThreadidx << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavetetthreadidx[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavebdrythreadidx[0]),
			numberofthreads
			);

		// Mark and append expanding(cutting) tets
		kernelUpdateCavity2StarShapedAppend << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_cavebdrylist[0]),
			thrust::raw_pointer_cast(&t_cavebdryprev[0]),
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdryhead[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavebdrythreadidx[0]),
			cavebdrystartindex,
			cavebdryexpandsize,
			thrust::raw_pointer_cast(&t_cavetetlist[0]),
			thrust::raw_pointer_cast(&t_cavetetprev[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetethead[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			thrust::raw_pointer_cast(&t_cavetetthreadidx[0]),
			cavetetstartindex,
			cavetetexpandsize,
			cavebdrycurstartindex,
			numberofthreads
			);

		kernelUpdateCavity2StarShapedUpdateListTails << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_cavethreadidx[0]),
			thrust::raw_pointer_cast(&t_cavetetnext[0]),
			thrust::raw_pointer_cast(&t_cavetettail[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavetetexpandindices[0]),
			cavetetstartindex,
			thrust::raw_pointer_cast(&t_cavebdrynext[0]),
			thrust::raw_pointer_cast(&t_cavebdrytail[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandsize[0]),
			thrust::raw_pointer_cast(&t_cavebdryexpandindices[0]),
			cavebdrystartindex,
			numberofthreads
			);

		if (drawmesh != NULL && drawmesh->animation)
		{
			// Output animation frame mesh
			if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
				iter_2 == drawmesh->iter_tet)
			{
				outputCavityFrame(
					drawmesh,
					t_pointlist,
					t_tetlist,
					t_tetmarker,
					t_threadmarker,
					t_caveoldtetlist,
					t_caveoldtetnext,
					t_caveoldtethead,
					iter_0,
					iter_1,
					iter_2,
					iteration,
					0
				);
			}
		}

		// Update working thread list
		numberofthreads = cavebdryexpandsize;
		iteration++;
		if (numberofthreads == 0)
			break;

		// Update variables
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		cavetetstartindex = t_cavetetlist.size();
		cavebdrycurstartindex = cavebdrystartindex;
		cavebdrystartindex = t_cavebdrylist.size();

		// Set threadidx list
		t_cavethreadidx.resize(numberofthreads);
		thrust::copy(t_cavebdrythreadidx.begin(), t_cavebdrythreadidx.end(), t_cavethreadidx.begin());

		// Reset expanding lists
		t_cavebdryexpandsize.resize(numberofthreads);
		thrust::fill(t_cavebdryexpandsize.begin(), t_cavebdryexpandsize.end(), 0);
		t_cavebdryexpandindices.resize(numberofthreads);

		t_cavetetexpandsize.resize(numberofthreads);
		thrust::fill(t_cavetetexpandsize.begin(), t_cavetetexpandsize.end(), 0);
		t_cavetetexpandindices.resize(numberofthreads);

		cudaDeviceSynchronize();
	}

	// Update the cavity boundary faces
	numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	if(debug_msg >= 2)
		printf("        After cutting cavity, numberofthreads = %d, total cutting iteration = %d\n", numberofthreads, iteration);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: cavity shrinking, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	kernelUpdateBoundaryFaces << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_cavetetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetnext[0]),
		thrust::raw_pointer_cast(&t_cavetethead[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdryprev[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_cavebdrytail[0]),
		thrust::raw_pointer_cast(&t_cutcount[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		numberofthreads
		);

	// Update the list of old tets
	kernelUpdateOldTets << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_insertidxlist[0]),
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_segstatus[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetprev[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_caveoldtettail[0]),
		thrust::raw_pointer_cast(&t_cutcount[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	// Check if cavities share boundaries
	kernelAdjacentCavitiesCheck << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_priority[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
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
	if (numberofthreads == 0)
		return 1;

	//printf("after adjacent cavities check, numberofthreads = %d\n", numberofthreads);

	if (t_caveshlist.size() != 0)
	{
		// Update the subcavities
		IntD t_cutshcount(numberofthreads, 0);
		kernelUpdateSubcavities << < numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2tetlist[0]),
			thrust::raw_pointer_cast(&t_caveshlist[0]),
			thrust::raw_pointer_cast(&t_caveshprev[0]),
			thrust::raw_pointer_cast(&t_caveshnext[0]),
			thrust::raw_pointer_cast(&t_caveshhead[0]),
			thrust::raw_pointer_cast(&t_caveshtail[0]),
			thrust::raw_pointer_cast(&t_cutshcount[0]),
			thrust::raw_pointer_cast(&t_tetmarker[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			numberofthreads
			);

		// Validity check
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
			thrust::raw_pointer_cast(&t_cutshcount[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		numberofthreads = updateActiveListByMarker_Slot(t_threadmarker, t_threadlist, t_threadmarker.size());
		numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
		if (numberofthreads == 0) // bad elements cause 0 valid cavities
			return 1;
	}

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Refinement elements check
	// The new point is inserted by Delaunay refinement, i.e., it is the 
	//   circumcenter of a tetrahedron, or a subface, or a segment.
	//   Do not insert this point if the tetrahedron, or subface, or segment
	//   is not inside the final cavity.
	kernelValidateRefinementElements << < numberofblocks, BLOCK_SIZE >> >(
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
	if(debug_msg >= 1)
		printf("        After validating cavity, numberofthreads = %d(#%d, #%d, #%d)\n", 
			numberofthreads,
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1),
			thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 2));

	//printf("%d, ", numberofthreads);

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: cavity validation, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	if (drawmesh != NULL && drawmesh->animation)
	{
		// Output animation frame mesh
		if (iter_0 == drawmesh->iter_seg && iter_1 == drawmesh->iter_subface &&
			iter_2 == drawmesh->iter_tet)
		{
			outputCavityFrame(
				drawmesh,
				t_pointlist,
				t_tetlist,
				t_tetmarker,
				t_threadmarker,
				t_caveoldtetlist,
				t_caveoldtetnext,
				t_caveoldtethead,
				iter_0,
				iter_1,
				iter_2,
				-1,
				0
			);
		}
	}

	if (drawmesh != NULL && !drawmesh->animation)
	{
		if (drawmesh->iter_seg != -1 || drawmesh->iter_subface != -1 || drawmesh->iter_tet != -1)
		{
			if (iter_0 == drawmesh->iter_seg  && iter_1 == drawmesh->iter_subface
				&& iter_2 == drawmesh->iter_tet)
			{
				printf("Copy mesh to drawmesh!\n");

				outputTmpMesh(
					drawmesh,
					t_pointlist, t_pointtypelist,
					t_seglist, t_segstatus,
					t_trifacelist, t_tristatus,
					t_tetlist, t_tetstatus,
					t_insertidxlist, t_insertptlist,
					t_threadlist, t_threadmarker,
					t_cavebdrylist, t_cavebdrynext, t_cavebdryhead,
					0);

				return 0;
			}
		}
	}


	if(iter_0 == -1 && iter_1 == -1 && iter_2 >= behavior->miniter)
	{
		if (numberofthreads < behavior->minthread)
			return 0;
	}

	if (numberofthreads == 0)
		return 1;

	// Get the length of the shortest edge connecting to new points
	RealD t_smlen(numofencsubseg + numofencsubface);
	IntD t_parentpt(numofencsubseg + numofencsubface);
	kernelComputeShortestEdgeLength << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_smlen[0]),
		thrust::raw_pointer_cast(&t_parentpt[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update cavity subseg list
	kernelUpdateCavitySubsegs << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegprev[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_cavetetsegtail[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_segmarker[0]),
		thrust::raw_pointer_cast(&t_segmarker2[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update cavity subface list
	kernelUpdateCavitySubfaces << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshprev[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_cavetetshtail[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		thrust::raw_pointer_cast(&t_trimarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Init point insertion, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Insert points into list
	oldsize = t_pointradius.size();
	int oldpointsize = oldsize;
	newsize = oldsize + numberofthreads;
	t_pointlist.resize(3 * newsize);
	t_point2trilist.resize(newsize, trihandle(-1, 0));
	t_point2tetlist.resize(newsize, tethandle(-1, 11));
	t_pointtypelist.resize(newsize);
	t_pointradius.resize(newsize, 0.0);

	kernelInsertNewPoints << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_pointtypelist[0]),
		thrust::raw_pointer_cast(&t_insertptlist[0]),
		thrust::raw_pointer_cast(&t_threadmarker[0]),
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

	// Create new tetrahedra to fill the cavity
	IntD t_tetexpandsize(numberofthreads, 0);
	IntD t_tetexpandindice(numberofthreads, -1);
	int tetexpandsize = 0;

	kernelCountNewTets << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetexpandsize[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	tetexpandsize = thrust::reduce(t_tetexpandsize.begin(), t_tetexpandsize.end());
	thrust::exclusive_scan(t_tetexpandsize.begin(), t_tetexpandsize.end(), t_tetexpandindice.begin());

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
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_point2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetexpandindice[0]),
		thrust::raw_pointer_cast(&t_emptyslot[0]),
		oldpointsize,
		numberofthreads
		);


	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Set a handle for speeding point location (may be not necessary)

	// Connect adjacent new tetrahedra together
	kernelConnectNewTetNeighbors << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_point2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		thrust::raw_pointer_cast(&t_tetmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Connect boundary subfaces to the new tets
	kernelConnectBoundarySubfaces2NewTets << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_trimarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Connect boundary segments to the new tets
	kernelConnectBoundarySubsegs2NewTets << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_segmarker[0]),
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[1] = clock();
		printf("        Stage: Insert new tets, time = %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Split a subface or a segment
	TriHandleD t_caveshbdlist;
	IntD t_caveshbdprev, t_caveshbdnext;
	IntD t_caveshbdhead(numofinsertpt, -1), t_caveshbdtail(numofinsertpt, -1);
	numberofsplittablesubsegs = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 0);
	numberofsplittablesubfaces = thrust::count(t_threadmarker.begin(), t_threadmarker.end(), 1);
	if ( numberofsplittablesubsegs != 0 || numberofsplittablesubfaces != 0)
	{
		// Create subcavity boundary edge list
		IntD t_caveshbdsize(numberofthreads, 0), t_caveshbdindices(numberofthreads, -1);
		kernelSubCavityBoundaryEdgeCheck << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_caveshlist[0]),
			thrust::raw_pointer_cast(&t_caveshnext[0]),
			thrust::raw_pointer_cast(&t_caveshhead[0]),
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

		int caveshbdsize = thrust::reduce(t_caveshbdsize.begin(), t_caveshbdsize.end());
		thrust::exclusive_scan(t_caveshbdsize.begin(), t_caveshbdsize.end(), t_caveshbdindices.begin());
		t_caveshbdlist.resize(caveshbdsize);
		t_caveshbdprev.resize(caveshbdsize);
		t_caveshbdnext.resize(caveshbdsize);

		kernelSubCavityBoundaryEdgeAppend << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_caveshlist[0]),
			thrust::raw_pointer_cast(&t_caveshnext[0]),
			thrust::raw_pointer_cast(&t_caveshhead[0]),
			thrust::raw_pointer_cast(&t_caveshbdlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdprev[0]),
			thrust::raw_pointer_cast(&t_caveshbdnext[0]),
			thrust::raw_pointer_cast(&t_caveshbdhead[0]),
			thrust::raw_pointer_cast(&t_caveshbdtail[0]),
			thrust::raw_pointer_cast(&t_caveshbdsize[0]),
			thrust::raw_pointer_cast(&t_caveshbdindices[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
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
			t_trimarker.resize(newsize, MAXULL); // extend to identify new subfaces when need to read from tristatus
			numberofemptyslot = updateEmptyTris(t_tristatus, t_emptyslot);
		}

		TriHandleD t_casout(caveshbdsize, trihandle(-1, 0)), t_casin(caveshbdsize, trihandle(-1, 0));

		kernelInsertNewSubfaces << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_point2trilist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tri2parentidxlist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_caveshbdlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdnext[0]),
			thrust::raw_pointer_cast(&t_caveshbdhead[0]),
			thrust::raw_pointer_cast(&t_caveshbdindices[0]),
			thrust::raw_pointer_cast(&t_emptyslot[0]),
			thrust::raw_pointer_cast(&t_casout[0]),
			thrust::raw_pointer_cast(&t_casin[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			oldpointsize,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		kernelConnectNewSubface2OuterSubface_Phase1 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_point2trilist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tri2parentidxlist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_caveshbdlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdnext[0]),
			thrust::raw_pointer_cast(&t_caveshbdhead[0]),
			thrust::raw_pointer_cast(&t_caveshbdindices[0]),
			thrust::raw_pointer_cast(&t_emptyslot[0]),
			thrust::raw_pointer_cast(&t_casout[0]),
			thrust::raw_pointer_cast(&t_casin[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			oldpointsize,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		kernelConnectNewSubface2OuterSubface_Phase2 << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_point2trilist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_seglist[0]),
			thrust::raw_pointer_cast(&t_seg2trilist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tri2parentidxlist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_caveshbdlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdnext[0]),
			thrust::raw_pointer_cast(&t_caveshbdhead[0]),
			thrust::raw_pointer_cast(&t_caveshbdindices[0]),
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

		kernelConnectNewSubfaceNeighbors << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_caveshbdlist[0]),
			thrust::raw_pointer_cast(&t_caveshbdnext[0]),
			thrust::raw_pointer_cast(&t_caveshbdhead[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		// Remove degenerated subfaces at segments
		kernelRemoveDegeneratedNewSubfaces << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_point2trilist[0]),
			thrust::raw_pointer_cast(&t_pointtypelist[0]),
			thrust::raw_pointer_cast(&t_trifacelist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_cavesegshlist[0]),
			thrust::raw_pointer_cast(&t_cavesegshnext[0]),
			thrust::raw_pointer_cast(&t_cavesegshhead[0]),
			thrust::raw_pointer_cast(&t_cavesegshtail[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			oldpointsize,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
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
				t_segmarker.resize(newsize, MAXINT); // extend to identify new subsegs when need to read from segstatus
				t_segencmarker.resize(newsize, -1);
				numberofemptyslot = updateEmptyTris(t_segstatus, t_emptyslot);
			}

			kernelInsertNewSubsegs << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_point2trilist[0]),
				thrust::raw_pointer_cast(&t_pointtypelist[0]),
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_seg2parentidxlist[0]),
				thrust::raw_pointer_cast(&t_segstatus[0]),
				thrust::raw_pointer_cast(&t_trifacelist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tri2seglist[0]),
				thrust::raw_pointer_cast(&t_segencmarker[0]),
				thrust::raw_pointer_cast(&t_cavesegshlist[0]),
				thrust::raw_pointer_cast(&t_cavesegshnext[0]),
				thrust::raw_pointer_cast(&t_cavesegshhead[0]),
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

			t_cavesegshlist.resize(2 * numberofsplittablesubsegs);
			t_cavesegshprev.resize(2 * numberofsplittablesubsegs);
			t_cavesegshnext.resize(2 * numberofsplittablesubsegs);
			thrust::fill(t_cavesegshhead.begin(), t_cavesegshhead.end(), -1);
			thrust::fill(t_cavesegshtail.begin(), t_cavesegshtail.end(), -1);
			// Connect new subsegs to outer subsegs and collect new subsegs into list
			kernelConnectNewSubseg2OuterSubseg << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_segmarker[0]),
				thrust::raw_pointer_cast(&t_cavesegshlist[0]),
				thrust::raw_pointer_cast(&t_cavesegshprev[0]),
				thrust::raw_pointer_cast(&t_cavesegshnext[0]),
				thrust::raw_pointer_cast(&t_cavesegshhead[0]),
				thrust::raw_pointer_cast(&t_cavesegshtail[0]),
				thrust::raw_pointer_cast(&t_emptyslot[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				numberofthreads
				);
		}

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		// Recover new subfaces in cavities
		kernelConnectNewSubfaces2NewTets << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
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
			thrust::raw_pointer_cast(&t_caveshbdnext[0]),
			thrust::raw_pointer_cast(&t_caveshbdhead[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			oldpointsize,
			numberofthreads
			);

		if (debug_error)
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		if (numberofsplittablesubsegs != 0)
		{
			t_randomseed.resize(numberofsplittablesubsegs); // Keep track of random seeds for all threads
			thrust::fill(t_randomseed.begin(), t_randomseed.end(), 1);

			kernelConnectNewSubsegs2NewTets << < numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_threadlist[0]),
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_point2tetlist[0]),
				thrust::raw_pointer_cast(&t_seglist[0]),
				thrust::raw_pointer_cast(&t_seg2trilist[0]),
				thrust::raw_pointer_cast(&t_seg2tetlist[0]),
				thrust::raw_pointer_cast(&t_tri2tetlist[0]),
				thrust::raw_pointer_cast(&t_tri2trilist[0]),
				thrust::raw_pointer_cast(&t_tetlist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tet2seglist[0]),
				thrust::raw_pointer_cast(&t_tetmarker[0]),
				thrust::raw_pointer_cast(&t_cavesegshlist[0]),
				thrust::raw_pointer_cast(&t_cavesegshnext[0]),
				thrust::raw_pointer_cast(&t_cavesegshhead[0]),
				thrust::raw_pointer_cast(&t_randomseed[0]),
				thrust::raw_pointer_cast(&t_threadmarker[0]),
				thrust::raw_pointer_cast(&t_insertidxlist[0]),
				numberofthreads
				);
		}

	}

	if (debug_timing)
	{
		cudaDeviceSynchronize();
		tv[0] = clock();
		printf("        Stage: Insert new subsegments and subfaces, time = %f\n", (REAL)(tv[0] - tv[1]));
	}

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update encroachment markers
	kernelUpdateSegencmarker << < numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_seg2tetlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_cavetetseglist[0]),
		thrust::raw_pointer_cast(&t_cavetetsegnext[0]),
		thrust::raw_pointer_cast(&t_cavetetseghead[0]),
		thrust::raw_pointer_cast(&t_cavesegshlist[0]),
		thrust::raw_pointer_cast(&t_cavesegshnext[0]),
		thrust::raw_pointer_cast(&t_cavesegshhead[0]),
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

	kernelUpdateSubfaceencmarker << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_tri2tetlist[0]),
		thrust::raw_pointer_cast(&t_tri2trilist[0]),
		thrust::raw_pointer_cast(&t_tristatus[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshlist[0]),
		thrust::raw_pointer_cast(&t_cavetetshnext[0]),
		thrust::raw_pointer_cast(&t_cavetetshhead[0]),
		thrust::raw_pointer_cast(&t_caveshbdlist[0]),
		thrust::raw_pointer_cast(&t_caveshbdnext[0]),
		thrust::raw_pointer_cast(&t_caveshbdhead[0]),
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

	// Update tet bad status
	kernelUpdateTetBadstatus << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_cavebdrylist[0]),
		thrust::raw_pointer_cast(&t_cavebdrynext[0]),
		thrust::raw_pointer_cast(&t_cavebdryhead[0]),
		behavior->radius_to_edge_ratio,
		numberofthreads
		);

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Update insertion radius after insertion
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
	}

	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	if (numberofsplittablesubsegs !=0 || numberofsplittablesubfaces != 0)
	{
		kernelResetOldSubfaceInfo << < numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_threadlist[0]),
			thrust::raw_pointer_cast(&t_tri2trilist[0]),
			thrust::raw_pointer_cast(&t_tri2seglist[0]),
			thrust::raw_pointer_cast(&t_tristatus[0]),
			thrust::raw_pointer_cast(&t_subfaceencmarker[0]),
			thrust::raw_pointer_cast(&t_caveshlist[0]),
			thrust::raw_pointer_cast(&t_caveshnext[0]),
			thrust::raw_pointer_cast(&t_caveshhead[0]),
			thrust::raw_pointer_cast(&t_threadmarker[0]),
			numberofthreads
			);
	}
	
	if (debug_error)
	{
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	kernelResetOldTetInfo << < numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_threadlist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tet2trilist[0]),
		thrust::raw_pointer_cast(&t_tet2seglist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_caveoldtetlist[0]),
		thrust::raw_pointer_cast(&t_caveoldtetnext[0]),
		thrust::raw_pointer_cast(&t_caveoldtethead[0]),
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
	}
#ifdef MYDEBUG
#endif // DEBUG

	// Release internal thrust arrays
	//t_tetmarker.clear();
	//t_tetmarker.shrink_to_fit();

	return 1;
}