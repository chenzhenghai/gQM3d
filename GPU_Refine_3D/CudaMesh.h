#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include "MeshStructure.h"
#include "CudaThrust.h"


#define REAL double
#define EPSILON 1.0e-8

#define BLOCK_SIZE 256

#define MAXINT 2147483647
#define MAXUINT 0xFFFFFFFF
#define MAXULL 0xFFFFFFFFFFFFFFFF
#define MAXFLT 3.402823466e+38F

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Helpers																	 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ uint64 cudamesh_encodeUInt64Priority(int priority, int index);

__device__ int cudamesh_getUInt64PriorityIndex(uint64 priority);

__device__ int cudamesh_getUInt64Priority(uint64 priority);

__device__ bool cudamesh_isNearZero(double val);


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric helpers														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ bool cudamesh_lu_decmp(REAL lu[4][4], int n, int* ps, REAL* d, int N);

__device__ void cudamesh_lu_solve(REAL lu[4][4], int n, int* ps, REAL* b, int N);

__device__ bool cudamesh_circumsphere(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL* cent, REAL* radius);

__device__ void cudamesh_facenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
	REAL* lav);

__device__ int cudamesh_segsegadjacent(
	int seg1,
	int seg2,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist
);

__device__ int cudamesh_segfacetadjacent(
	int subseg,
	int subsh,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist
);

__device__ int cudamesh_facetfacetadjacent(
	int subsh1,
	int subsh2,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist
);

__device__ REAL cudamesh_triangle_squared_area(
	REAL* pa, REAL* pb, REAL* pc
);

__device__ REAL cudamesh_tetrahedronvolume(
	REAL* pa, REAL* pb, REAL* pc, REAL* pd
);

__device__ REAL cudamesh_tetrahedronvolume(
	int tetid,
	REAL* d_pointlist,
	int* d_tetlist
);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric predicates														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ REAL cudamesh_insphere_s(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe,
	int ia, int ib, int ic, int id, int ie);

__device__ REAL cudamesh_incircle3d(REAL* pa, REAL* pb, REAL* pc, REAL* pd);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Mesh manipulation primitives                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Init fast lookup tables */
void cudamesh_inittables();

/* Init bounding box*/
void cudamesh_initbbox(
	int numofpoints, double* pointlist,
	int& xmax, int& xmin, int& ymax, int& ymin, int& zmax, int& zmin);

/* Init Geometric primitives */
void cudamesh_exactinit(int verbose, int noexact, int nofilter, 
	REAL maxx, REAL maxy, REAL maxz);

/* Init Kernel constants */
void cudamesh_initkernelconstants(REAL maxx, REAL maxy, REAL maxz);

/* Primitives for points */
// Convert point index to pointer to pointlist
__device__ double* cudamesh_id2pointlist(int index, double* pointlist);

/* Primitives for tetrahedron */

__device__ int cudamesh_org(tethandle t, int* tetlist);
__device__ int cudamesh_dest(tethandle t, int* tetlist);
__device__ int cudamesh_apex(tethandle t, int* tetlist);
__device__ int cudamesh_oppo(tethandle t, int* tetlist);
__device__ void cudamesh_setorg(tethandle t, int p, int* tetlist);
__device__ void cudamesh_setdest(tethandle t, int p, int* tetlist);
__device__ void cudamesh_setapex(tethandle t, int p, int* tetlist);
__device__ void cudamesh_setoppo(tethandle t, int p, int* tetlist);

__device__ void cudamesh_bond(tethandle t1, tethandle t2, tethandle* neighborlist);
__device__ void cudamesh_dissolve(tethandle t, tethandle* neighborlist);

__device__ void cudamesh_esym(tethandle& t1, tethandle& t2);
__device__ void cudamesh_esymself(tethandle& t);
__device__ void cudamesh_enext(tethandle& t1, tethandle& t2);
__device__ void cudamesh_enextself(tethandle& t);
__device__ void cudamesh_eprev(tethandle& t1, tethandle& t2);
__device__ void cudamesh_eprevself(tethandle& t);
__device__ void cudamesh_enextesym(tethandle& t1, tethandle& t2);
__device__ void cudamesh_enextesymself(tethandle& t);
__device__ void cudamesh_eprevesym(tethandle& t1, tethandle& t2);
__device__ void cudamesh_eprevesymself(tethandle& t);
__device__ void cudamesh_eorgoppo(tethandle& t1, tethandle& t2);
__device__ void cudamesh_eorgoppoself(tethandle& t);
__device__ void cudamesh_edestoppo(tethandle& t1, tethandle& t2);
__device__ void cudamesh_edestoppoself(tethandle& t);

__device__ void cudamesh_fsym(tethandle& t1, tethandle& t2, tethandle* neighborlist);
__device__ void cudamesh_fsymself(tethandle& t, tethandle* neighborlist);
__device__ void cudamesh_fnext(tethandle& t1, tethandle& t2, tethandle* neigenhborlist);
__device__ void cudamesh_fnextself(tethandle& t, tethandle* neighborlist);

__device__ bool cudamesh_ishulltet(tethandle t, int* tetlist);
__device__ bool cudamesh_isdeadtet(tethandle t);

// Primitives for subfaces and subsegments.
__device__ void cudamesh_spivot(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
__device__ void cudamesh_spivotself(trihandle& s, trihandle* tri2trilist);
__device__ void cudamesh_sbond(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
__device__ void cudamesh_sbond1(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
__device__ void cudamesh_sdissolve(trihandle& s, trihandle* tri2trilist);
__device__ int cudamesh_sorg(trihandle& s, int* trilist);
__device__ int cudamesh_sdest(trihandle& s, int* trilist);
__device__ int cudamesh_sapex(trihandle& s, int* trilist);
__device__ void cudamesh_setsorg(trihandle& s, int p, int* trilist);
__device__ void cudamesh_setsdest(trihandle& s, int p, int* trilist);
__device__ void cudamesh_setsapex(trihandle& s, int p, int* trilist);
__device__ void cudamesh_sesym(trihandle& s1, trihandle& s2);
__device__ void cudamesh_sesymself(trihandle& s);
__device__ void cudamesh_senext(trihandle& s1, trihandle& s2);
__device__ void cudamesh_senextself(trihandle& s);
__device__ void cudamesh_senext2(trihandle& s1, trihandle& s2);
__device__ void cudamesh_senext2self(trihandle& s);

// Primitives for interacting tetrahedra and subfaces.
__device__ void cudamesh_tsbond(tethandle& t, trihandle& s, trihandle* tet2trilist, tethandle* tri2tetlist);
__device__ void cudamesh_tspivot(tethandle& t, trihandle& s, trihandle* tet2trilist);
__device__ void cudamesh_stpivot(trihandle& s, tethandle& t, tethandle* tri2tetlist);

// Primitives for interacting tetrahedra and segments.
__device__ void cudamesh_tsspivot1(tethandle& t, trihandle& seg, trihandle* tet2seglist);
__device__ void cudamesh_tssbond1(tethandle& t, trihandle& seg, trihandle* tet2seglist);
__device__ void cudamesh_sstbond1(trihandle& s, tethandle& t, tethandle* seg2tetlist);
__device__ void cudamesh_sstpivot1(trihandle& s, tethandle& t, tethandle* seg2tetlist);

// Primitives for interacting subfaces and segments.
__device__ void cudamesh_ssbond(trihandle& s, trihandle& edge, trihandle* tri2seglist, trihandle* seg2trilist);
__device__ void cudamesh_ssbond1(trihandle& s, trihandle& edge, trihandle* tri2seglist);
__device__ void cudamesh_sspivot(trihandle& s, trihandle& edge, trihandle* tri2seglist);
__device__ bool cudamesh_isshsubseg(trihandle&s, trihandle* tri2seglist);

/* Advanced primitives. */
__device__ void cudamesh_point2tetorg(int pa, tethandle& searchtet, tethandle* point2tetlist, int* tetlist);

/* Geometric calculations (non-robust) */
__device__ REAL cudamesh_dot(REAL* v1, REAL* v2);
__device__ REAL cudamesh_distance(REAL* p1, REAL* p2);
__device__ void cudamesh_cross(REAL* v1, REAL* v2, REAL* n);

/* Helpers */
__device__ unsigned long cudamesh_randomnation(unsigned long* randomseed, unsigned int choices);
__device__ enum interresult cudamesh_finddirection(tethandle* searchtet, int endpt, double* pointlist, int* tetlist, tethandle* neighborlist, unsigned long* randomseed);
//int getedge(int e1, int e2, tethandle *tedge, tethandle* point2tet, double* pointlist, int* tetlist, tethandle* neighborlist, int* markerlist);

/* Refinement */

// Insertion radius

// Insert point
__global__ void kernelCheckAbortiveElements(
	int* d_insertidxlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int insertiontype,
	int numofinsertpt
);

__global__ void kernelCheckInsertRadius_Seg(
	int* d_segidlist,
	REAL* d_pointlist,
	REAL* d_pointradius,
	int* d_seglist,
	tristatus* d_segstatus,
	int* d_segencmarker,
	int* d_threadmarker,
	int numofseg
);

__global__ void kernelComputePriority_Seg(
	int* d_segidlist,
	int* d_threadlist,
	int* d_seglist,
	REAL* d_pointlist,
	int* d_priority,
	int numofthreads
);

__global__ void kernelInitSearchTet_Seg(
	int* d_segidlist,
	int* d_threadlist,
	tethandle* d_seg2tetlist,
	tethandle* d_searchtetlist,
	int numofthreads
);

__global__ void kernelCheckInsertRadius_Subface(
	int* d_subfaceidlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	REAL* d_pointradius,
	int* d_threadmarker,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	tristatus* d_tristatus,
	int* d_subfaceencmarker,
	int numofthreads
);

__global__ void kernelInitSearchshList(
	int* d_subfaceidlist,
	int* d_threadlist,
	trihandle* d_searchsh,
	int numofthreads
);

__global__ void kernelSurfacePointLocation(
	int* d_subfaceidlist,
	trihandle* d_searchsh,
	tethandle* d_searchtetlist,
	locateresult* d_pointlocation,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	int* d_threadlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	unsigned long* d_randomseed,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelComputePriority_Subface(
	int* d_insertidxlist,
	int* d_threadlist,
	int* d_trifacelist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	REAL* d_pointlist,
	int* d_priority,
	int numofthreads
);

__global__ void kernelCheckInsertRadius_Tet(
	int* d_tetidlist,
	REAL* d_pointlist,
	REAL* d_pointradius,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelPointLocation(
	int* d_tetidlist,
	REAL* d_insertptlist,
	locateresult* d_pointlocation,
	tethandle* d_searchtetlist,
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_priority,
	unsigned long* d_randomseed,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelComputeSteinerPoint_Tet(
	int* d_tetidlist,
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_priority,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelMarkInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	trihandle* d_searchsh,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tethandle* d_neighborlist,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_segmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int insertiontype,
	int numofthreads
);

__global__ void kernelCompactSteinerPoints(
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_tmplist,
	int numofthreads
);

__global__ void kernelCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	trihandle* d_searchsh,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tethandle* d_neighborlist,
	int* d_initialcavitysize,
	int* d_initialsubcavitysize,
	int insertiontype,
	int numofthreads
);

__global__ void kernelMarkAndCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	trihandle* d_searchsh,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tethandle* d_neighborlist,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_segmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int* d_initialsubcavitysize,
	int numofthreads
);

__global__ void kernelMarkAndCountInitialCavity(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	trihandle* d_searchsh,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tethandle* d_neighborlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_segmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int* d_initialsubcavitysize,
	int numofthreads
);

__global__ void kernelCheckRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int* d_initialcavitysize,
	int* d_initialsubcavitysize,
	int numofencsubseg,
	int numofencsubface,
	int numofbadelement,
	int numofthreads
);

__global__ void kernelKeepRecordOldtet(
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelInitCavityLinklist(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	trihandle* d_searchsh,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_initialcavityindices,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_initialsubcavityindices,
	int* d_initialsubcavitysize,
	int* d_cavethreadidx,
	trihandle* d_caveshlist,
	int* d_caveshprev,
	int* d_caveshnext,
	int* d_caveshhead,
	int* d_caveshtail,
	trihandle* d_cavesegshlist,
	int* d_cavesegshprev,
	int* d_cavesegshnext,
	int* d_cavesegshhead,
	int* d_cavesegshtail,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelInitCavityLinklist(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	tethandle* d_searchtet,
	trihandle* d_searchsh,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_initialcavityindices,
	int* d_initialcavitysize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_cavetetlist,
	int* d_cavetetidx,
	int* d_initialsubcavityindices,
	int* d_initialsubcavitysize,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	trihandle* d_cavesegshlist,
	int* d_cavesegshidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSetReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int oldcaveoldtetsize,
	int numofthreads
);

__global__ void kernelCheckCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandsize,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int numofthreads
);

__global__ void kernelAppendCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_cavetetlist,
	int* d_cavetetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandindices,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int oldcavetetsize,
	int numofthreads
);

__global__ void kernelInitLinklistCurPointer(
	int* d_threadlist,
	int* d_linklisthead,
	int* d_linklistcur,
	int numofthreads
);

__global__ void kernelCavityRatioControl(
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavethreadidx,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavetetidx,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelMarkCavityReuse(
	int* d_insertidxlist,
	int* d_cavetetidx,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelMarkOldtetlist(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSetRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int oldrecordsize,
	int numofthreads
);

__global__ void kernelMarkLargeCavityAsLoser(
	int* d_cavetetidx,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingCheck(
	int* d_cavethreadidx,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingCheck(
	int* d_cavetetidx,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavetetlist,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void  kernelCorrectExpandingSize(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int numofthreads
);

__global__ void  kernelCorrectExpandingSize(
	int* d_cavetetidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingSetThreadidx(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int* d_caveoldtetthreadidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int numofthreads
);

__global__ void kernelCavityExpandingMarkAndAppend(
	int* d_cavethreadidx,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int cavetetstartindex,
	int cavetetexpandsize,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int* d_caveoldtetthreadidx,
	int caveoldtetstartindex,
	int caveoldtetexpandsize,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int cavebdrystartindex,
	int cavebdryexpandsize,
	int* d_threadmarker,
	int* d_priority,
	uint64* d_tetmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingMarkAndAppend(
	int* d_cavetetidx,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int caveoldtetstartindex,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
);

__global__ void kernelCavityExpandingUpdateListTails(
	int* d_cavethreadidx,
	int* d_cavetetnext,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	int* d_caveoldtetnext,
	int* d_caveoldtettail,
	int* d_caveoldtetexpandsize,
	int* d_caveoldtetexpandindices,
	int caveoldtetstartindex,
	int* d_cavebdrynext,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_threadmarker,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelMarkCavityAdjacentSubsegs(
	int* d_threadlist,
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_segmarker,
	int* d_threadmarker,
	int numofthreads,
	uint64* d_tetmarker
);

__global__ void kernelMarkCavityAdjacentSubsegs(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_priority,
	uint64* d_segmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountCavitySubsegs(
	int* d_threadlist,
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_cavetetsegsize,
	int* d_segmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountCavitySubsegs_Phase1(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	uint64* d_segmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountCavitySubsegs_Phase2(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_cavetetsegsize,
	uint64* d_segmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelAppendCavitySubsegs(
	int* d_threadlist,
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegprev,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	int* d_cavetetsegtail,
	int* d_cavetetsegsize,
	int* d_cavetetsegindices,
	int* d_segmarker,
	int numofthreads
);

__global__ void kernelAppendCavitySubsegs(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	int* d_cavetetsegsize,
	int* d_cavetetsegindices,
	uint64* d_segmarker,
	int numofthreads
);

__global__ void kernelCheckSegmentEncroachment(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_seglist,
	int* d_segencmarker,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckSegmentEncroachment_Phase1(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	REAL* d_pointlist,
	int* d_seglist,
	int* d_segencmarker,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckSegmentEncroachment_Phase2(
	int* d_insertidxlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_cavetetsegidx,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelMarkCavityAdjacentSubfaces(
	int* d_threadlist,
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelMarkCavityAdjacentFaces(
	trihandle* d_tet2trilist,
	tethandle* d_neighborlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountCavitySubfaces(
	int* d_threadlist,
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_cavetetshsize,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountCavitySubfaces_Phase1(
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountCavitySubfaces_Phase2(
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_cavetetshsize,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelAppendCavitySubfaces(
	int* d_threadlist,
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	trihandle* d_cavetetshlist,
	int* d_cavetetshprev,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_cavetetshtail,
	int* d_cavetetshsize,
	int* d_cavetetshindices,
	int* d_trimarker,
	int numofthreads
);

__global__ void kernelAppendCavitySubfaces(
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	int* d_cavetetshsize,
	int* d_cavetetshindices,
	int* d_trimarker,
	int numofthreads
);

__global__ void kernelCheckSubfaceEncroachment(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	locateresult* d_pointlocation,
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_trifacelist,
	int* d_subfaceencmarker,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckSubfaceEncroachment_Phase1(
	REAL* d_insertptlist,
	REAL* d_pointlist,
	int* d_trifacelist,
	int* d_subfaceencmarker,
	tristatus* d_tristatus,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckSubfaceEncroachment_Phase2(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	tetstatus* d_tetstatus,
	int* d_cavetetshidx,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSubCavityExpandingCheck(
	int* d_threadlist,
	REAL* d_pointlist,
	tethandle* d_neighborlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist, 
	trihandle* d_tri2seglist,
	REAL* d_insertptlist,
	trihandle* d_caveshlist,
	int* d_caveshcur,
	int* d_caveshexpandsize,
	int* d_caveshexpandflag,
	int* d_priority,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int numofthreads
);

__global__ void kernelSubCavityExpandingCheck(
	REAL* d_pointlist,
	tethandle* d_neighborlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	REAL* d_insertptlist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	int* d_caveshexpandsize,
	int* d_caveshexpandflag,
	int* d_priority,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int caveshcurstartindex,
	int numofthreads
);

__global__ void kernelSubCavityExpandingAppend(
	int* d_threadlist,
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshprev,
	int* d_caveshnext,
	int* d_caveshhead,
	int* d_caveshtail,
	int* d_caveshcur,
	int* d_caveshexpandsize,
	int* d_caveshexpandindices,
	int* d_caveshexpandflag,
	int caveshstartindex,
	int* d_threadfinishmarker,
	int numofthreads
);

__global__ void kernelSubCavityExpandingAppend(
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	int* d_caveshexpandsize,
	int* d_caveshexpandindices,
	int* d_caveshexpandflag,
	int caveshcurstartindex,
	int caveshstartindex,
	int numofthreads
);

__global__ void kernelCavityBoundarySubfacesCheck(
	int* d_insertidxlist,
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tethandle* d_tri2tetlist,
	REAL* d_insertptlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_cavetetshmarker,
	tethandle* d_cavetetshflag,
	int* d_cavebdryexpandsize,
	int* d_cutcount,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int numofthreads
);

__global__ void kernelCavityBoundarySubfacesCheck(
	int* d_insertidxlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tethandle* d_tri2tetlist,
	REAL* d_insertptlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	tethandle* d_cavetetshflag,
	int* d_cavebdryexpandsize,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCavityBoundarySubfacesAppend(
	int* d_threadlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_cavetetshmarker,
	tethandle* d_cavetetshflag,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelCavityBoundarySubfacesAppend(
	int* d_cavetetshidx,
	tethandle* d_cavetetshflag,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelCavityBoundarySubsegsCheck(
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	REAL* d_insertptlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	int* d_cavetetsegmarker,
	tethandle* d_cavetetsegflag,
	int* d_cavebdryexpandsize,
	int* d_cutcount,
	uint64* d_tetmarker,
	int* d_segmarker,
	int numofthreads
);

__global__ void kernelCavityBoundarySubsegsCheck(
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	REAL* d_insertptlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	tethandle* d_cavetetsegflag,
	int* d_cavebdryexpandsize,
	uint64* d_tetmarker,
	int* d_segmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCavityBoundarySubsegsAppend(
	int* d_threadlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	int* d_cavetetsegmarker,
	tethandle* d_cavetetsegflag,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelCavityBoundarySubsegsAppend(
	int* d_cavetetsegidx,
	tethandle* d_cavetetsegflag,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedSortOutBoundaryListCount(
	int* d_threadlist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavecount,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedSortOutBoundaryListAppend(
	int* d_threadlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	tethandle* d_cavelist,
	int* d_caveprev,
	int* d_cavenext,
	int* d_expandindices,
	int* d_cavethreadidx,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedCheck(
	int* d_insertidxlist,
	int* d_cavethreadidx,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2seglist,
	REAL* d_insertptlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavetetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_cutcount,
	uint64* d_tetmarker,
	int cavebdrycurstartindex,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedCheck(
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	REAL* d_insertptlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int cavebdrycurstartindex,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedSetThreadidx(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedAppend(
	int* d_cavethreadidx,
	tethandle* d_neighborlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int cavebdrystartindex,
	int cavebdryexpandsize,
	tethandle* d_cavetetlist,
	int* d_cavetetprev,
	int* d_cavetetnext,
	int* d_cavetethead,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int cavetetstartindex,
	int cavetetexpandsize,
	int cavebdrycurstartindex,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedAppend(
	tethandle* d_neighborlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int cavebdrycurstartindex,
	int numofthreads
);

__global__ void kernelUpdateCavity2StarShapedUpdateListTails(
	int* d_cavethreadidx,
	int* d_cavetetnext,
	int* d_cavetettail,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int cavetetstartindex,
	int* d_cavebdrynext,
	int* d_cavebdrytail,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
);

__global__ void kernelUpdateBoundaryFaces(
	int* d_threadlist,
	tethandle* d_neighborlist,
	tethandle* d_cavetetlist,
	int* d_cavetetnext,
	int* d_cavetethead,
	tethandle* d_cavebdrylist,
	int* d_cavebdryprev,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavebdrytail,
	int* d_cutcount,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelUpdateBoundaryFaces(
	tethandle* d_neighborlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateOldTets(
	int* d_insertidxlist,
	int* d_threadlist,
	tethandle* d_neighborlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetprev,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_caveoldtettail,
	int* d_cutcount,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateOldTets(
	tethandle* d_neighborlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelAdjacentCavitiesCheck(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	int* d_priority,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateSubcavities(
	int* d_threadlist,
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	trihandle* d_caveshlist,
	int* d_caveshprev,
	int* d_caveshnext,
	int* d_caveshhead,
	int* d_caveshtail,
	int* d_cutshcount,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int numofthreads
);

__global__ void kernelUpdateSubcavities(
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelValidateSubcavities(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	trihandle* d_searchsh,
	int* d_threadlist,
	trihandle* d_seg2trilist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	int* d_segmarker,
	uint64* d_trimarker,
	int* d_cutshcount,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelValidateSubcavities(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	trihandle* d_searchsh,
	int* d_threadlist,
	trihandle* d_seg2trilist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	int* d_segmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelValidateRefinementElements(
	int* d_insertidxlist,
	trihandle* d_searchsh,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	int* d_threadlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	uint64* d_trimarker,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelValidateRefinementElements_New(
	int* d_insertidxlist,
	trihandle* d_searchsh,
	tethandle* d_searchtet,
	tethandle* d_neighborlist,
	int* d_threadlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	uint64* d_trimarker,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckDistances2ClosePoints(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	locateresult* d_pointlocation,
	tethandle* d_searchtet,
	int* d_threadlist,
	REAL* d_pointlist,
	tethandle* d_neighborlist,
	int* d_tetlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelResetCavityReuse(
	int* d_insertidxlist,
	int* d_threadlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelComputeShortestEdgeLength(
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	REAL* d_insertptlist,
	REAL* d_smlen,
	int* d_parentpt,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateCavitySubsegs(
	int* d_threadlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegprev,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	int* d_cavetetsegtail,
	uint64* d_tetmarker,
	int* d_segmarker,
	int* d_segmarker2,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateCavitySubsegs(
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	uint64* d_tetmarker,
	int* d_segmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateCavitySubfaces(
	int* d_threadlist,
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshprev,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	int* d_cavetetshtail,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int numofthreads
);

__global__ void kernelUpdateCavitySubfaces(
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelSetCavityThreadIdx(
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelComputeShortestEdgeLength_Phase1(
	int* d_cavebdryidx,
	int* d_scanleft,
	int* d_scanright,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelComputeShortestEdgeLength_Phase2(
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tethandle* d_cavebdrylist,
	REAL* d_insertptlist,
	REAL* d_smlen,
	int* d_parentpt,
	int* d_scanleft,
	int* d_scanright,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCountNewTets(
	int* d_threadlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandsize,
	int numofthreads
);

__global__ void kernelInsertNewTets(
	int* d_threadlist,
	tethandle* d_point2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandindice,
	int* d_emptytetindice,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewTets(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	tethandle* d_point2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int* d_emptytetindices,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewTetNeighbors(
	int* d_threadlist,
	tethandle* d_point2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelConnectNewTetNeighbors(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	tethandle* d_point2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	uint64* d_tetmarker,
	int numofthreads
);

__global__ void kernelConnectBoundarySubfaces2NewTets(
	int* d_threadlist,
	tethandle* d_tri2tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	uint64* d_trimarker,
	int numofthreads
);

__global__ void kernelConnectBoundarySubfaces2NewTets(
	tethandle* d_tri2tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	uint64* d_trimarker,
	int numofthreads
);

__global__ void kernelConnectBoundarySubsegs2NewTets(
	int* d_threadlist,
	tethandle* d_seg2tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	int* d_segmarker,
	int numofthreads
);

__global__ void kernelConnectBoundarySubsegs2NewTets(
	tethandle* d_seg2tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2seglist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	int* d_segmarker,
	int numofthreads
);

__global__ void kernelSubCavityBoundaryEdgeCheck(
	int* d_threadlist,
	trihandle* d_tri2seglist,
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshnext,
	int* d_caveshhead,
	int* d_caveshbdsize,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSubCavityBoundaryEdgeCheck(
	trihandle* d_tri2seglist,
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	int* d_caveshbdsize,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSubCavityBoundaryEdgeAppend(
	int* d_threadlist,
	trihandle* d_tri2seglist,
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshnext,
	int* d_caveshhead,
	trihandle* d_caveshbdlist,
	int* d_caveshbdprev,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	int* d_caveshbdtail,
	int* d_caveshbdsize,
	int* d_caveshbdindice,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelSubCavityBoundaryEdgeAppend(
	trihandle* d_tri2seglist,
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	int* d_caveshbdsize,
	int* d_caveshbdindices,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelInsertNewSubfaces(
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	int* d_tri2parentidxlist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	int* d_caveshbdindices,
	int* d_emptytriindices,
	trihandle* d_casout,
	trihandle* d_casin,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewSubfaces(
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	int* d_tri2parentidxlist,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	int* d_emptytriindices,
	trihandle* d_casout,
	trihandle* d_casin,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubface2OuterSubface_Phase1(
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	int* d_tri2parentidxlist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	int* d_caveshbdindices,
	int* d_emptytriindices,
	trihandle* d_casout,
	trihandle* d_casin,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubface2OuterSubface_Phase1(
	int* d_seglist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	int* d_emptytriindices,
	trihandle* d_casout,
	trihandle* d_casin,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelConnectNewSubface2OuterSubface_Phase2(
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	int* d_tri2parentidxlist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	int* d_caveshbdindices,
	int* d_emptytriindices,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubface2OuterSubface_Phase2(
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	int* d_emptytriindices,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubfaceNeighbors(
	int* d_threadlist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelConnectNewSubfaceNeighbors(
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelRemoveDegeneratedNewSubfaces(
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	trihandle* d_cavesegshlist,
	int* d_cavesegshnext,
	int* d_cavesegshhead,
	int* d_cavesegshtail,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelRemoveDegeneratedNewSubfaces(
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	trihandle* d_cavesegshlist,
	int* d_cavesegshidx,
	int* d_initialsubcavitysize,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewSubsegs(
	int* d_segidlist,
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	int* d_seg2parentidxlist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	int* d_segencmarker,
	trihandle* d_cavesegshlist,
	int* d_cavesegshnext,
	int* d_cavesegshhead,
	int* d_emptysegindices,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelInsertNewSubsegs(
	int* d_segidlist,
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	int* d_seg2parentidxlist,
	tristatus* d_segstatus,
	int* d_segencmarker,
	int* d_emptysegindices,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubseg2NewSubface(
	int* d_segidlist,
	int* d_threadlist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	trihandle* d_cavesegshlist,
	int* d_cavesegshidx,
	int* d_emptysegindices,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubseg2OuterSubseg(
	int* d_segidlist,
	int* d_threadlist,
	trihandle* d_seg2trilist,
	int* d_segmarker,
	trihandle* d_cavesegshlist,
	int* d_cavesegshprev,
	int* d_cavesegshnext,
	int* d_cavesegshhead,
	int* d_cavesegshtail,
	int* d_emptysegindices,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelConnectNewSubseg2OuterSubseg(
	int* d_segidlist,
	int* d_threadlist,
	trihandle* d_seg2trilist,
	int* d_segmarker,
	trihandle* d_cavesegshlist,
	int* d_cavesegshidx,
	int* d_emptysegindices,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelConnectNewSubfaces2NewTets(
	int* d_threadlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	uint64* d_tetmarker,
	trihandle* d_caveshbdlist,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubfaces2NewTets(
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	uint64* d_tetmarker,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelConnectNewSubsegs2NewTets(
	int* d_threadlist,
	REAL* d_pointlist,
	tethandle* d_point2tetlist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	tethandle* d_seg2tetlist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2seglist,
	uint64* d_tetmarker,
	trihandle* d_cavesegshlist,
	int* d_cavesegshnext,
	int* d_cavesegshhead,
	unsigned long* d_randomseed,
	int* d_threadmarker,
	int* d_insertidx,
	int numofthreads
);

__global__ void kernelConnectNewSubsegs2NewTets(
	REAL* d_pointlist,
	tethandle* d_point2tetlist,
	int* d_seglist,
	trihandle* d_seg2trilist,
	tethandle* d_seg2tetlist,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2seglist,
	trihandle* d_cavesegshlist,
	int* d_cavesegshidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelResetOldSubsegInfo(
	int* d_segidlist,
	int* d_threadlist,
	int* d_threadmarker,
	trihandle* d_seg2trilist,
	int numofthreads
);

__global__ void kernelResetOldSubfaceInfo(
	int* d_threadlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_subfaceencmarker,
	trihandle* d_caveshlist,
	int* d_caveshnext,
	int* d_caveshhead,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelResetOldSubfaceInfo(
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_subfaceencmarker,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelResetOldTetInfo(
	int* d_threadlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int numofthreads
);

__global__ void kernelResetOldTetInfo(
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int numofthreads
);

__global__ void kernelUpdateSegencmarker(
	int* d_threadlist,
	REAL * d_pointlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegnext,
	int* d_cavetetseghead,
	trihandle* d_cavesegshlist,
	int* d_cavesegshnext,
	int* d_cavesegshhead,
	int* d_segmarker,
	int* d_segencmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateSegencmarker_Phase1(
	REAL * d_pointlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	int* d_segmarker,
	int* d_segencmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateSegencmarker_Phase2(
	REAL * d_pointlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_cavesegshlist,
	int* d_cavesegshidx,
	int* d_segmarker,
	int* d_segencmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateSubfaceencmarker(
	int* d_threadlist,
	REAL * d_pointlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshnext,
	int* d_cavetetshhead,
	trihandle* d_caveshbdlist,
	int* d_caveshbdnext,
	int* d_caveshbdhead,
	uint64* d_trimarker,
	int* d_subfaceencmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateSubfaceencmarker_Phase1(
	REAL * d_pointlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	int* d_tetlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	uint64* d_trimarker,
	int* d_subfaceencmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateSubfaceencmarker_Phase2(
	REAL * d_pointlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	int* d_subfaceencmarker,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelUpdateTetBadstatus(
	int* d_threadlist,
	REAL* d_pointlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	REAL minratio,
	int numofthreads
);

__global__ void kernelUpdateTetBadstatus(
	REAL* d_pointlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	REAL minratio,
	int numofthreads
);

__global__ void kernelUpdateInsertRadius(
	int* d_threadlist,
	int* d_insertidxlist,
	REAL* d_pointlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	REAL* d_pointradius,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	int* d_tetlist,
	REAL* d_smlen,
	int* d_parentpt,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelUpdateInsertRadius(
	int* d_threadlist,
	int* d_insertidxlist,
	REAL* d_pointlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	REAL* d_pointradius,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	int* d_tetlist,
	uint64* d_smlenparentpt,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelUpdateInsertRadius_Seg(
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	REAL* d_pointradius,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	REAL* d_smlen,
	int* d_parentpt,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelUpdateInsertRadius_Subface(
	int* d_threadlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	REAL* d_pointradius,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	REAL* d_smlen,
	int* d_parentpt,
	int oldpointsize,
	int numofthreads
);

__global__ void kernelUpdateInsertRadius_Tet(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_pointlist,
	REAL* d_pointradius,
	int* d_tetlist,
	int oldpointsize,
	int numofthreads
);

// Check mesh
__global__ void kernelCheckPointNeighbors(
	trihandle* d_point2trilist,
	tethandle* d_point2tetlist,
	verttype* d_pointtypelist,
	int* d_seglist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelCheckSubsegNeighbors(
	int* d_seglist,
	trihandle* d_seg2trilist,
	tethandle* d_seg2tetlist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelCheckSubfaceNeighbors(
	int* d_seglist,
	trihandle* d_seg2trilist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	trihandle* d_tet2trilist,
	tetstatus* d_tetstatus,
	int numofthreads
);

__global__ void kernelCheckTetNeighbors(
	int* d_seglist,
	tethandle* d_seg2tetlist,
	tristatus* d_segstatus,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	int numofthreads
);

// Split bad element
__global__ void kernelCheckBadElementList(
	int* d_badeleidlist,
	int* d_threadmarker,
	int* d_segencmarker,
	int* d_subfaceencmarker,
	tetstatus* d_tetstatus,
	int numofencsegs,
	int numofencsubfaces,
	int numofbadtets,
	int numofthreads
);

__global__ void kernelComputeSteinerPointAndPriority(
	REAL* d_pointlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	int* d_seg2parentlist,
	int* d_segparentlist,
	int* d_segencmarker,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	REAL* d_priority,
	int numofthreads
);

__global__ void kernelComputePriorities(
	REAL* d_pointlist,
	int* d_seglist,
	int* d_trifacelist,
	int* d_tetlist,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_priority,
	int numofthreads
);

__global__ void kernelComputeSteinerPoints(
	REAL* d_pointlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	int* d_seg2parentlist,
	int* d_segparentlist,
	int* d_segencmarker,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	int numofthreads
);

__global__ void kernelModifyPriority(
	REAL* d_priorityreal,
	int* d_priorityint,
	REAL offset0,
	REAL offset1,
	REAL offset2,
	int* d_threadmarker,
	int numofthreads
);

__global__ void kernelCheckInsertRadius(
	REAL* d_pointlist,
	REAL* d_pointradius,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	tristatus* d_segstatus,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_segencmarker,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist,
	int* d_subfaceencmarker,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_steinerptlist,
	int numofthreads
);

__global__ void kernelLocatePoint(
	REAL* d_pointlist,
	tethandle* d_seg2tetlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_priority,
	unsigned long* d_randomseed,
	locateresult* d_pointlocation,
	trihandle* d_searchsh,
	tethandle* d_searchtet,
	int* d_insertidxlist,
	int* d_threadmarker,
	int* d_threadlist,
	REAL* d_steinerptlist,
	int numofsplittablesubsegs,
	int numofthreads
);

__global__ void kernelLocatePoint(
	REAL* d_pointlist,
	tethandle* d_seg2tetlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	int* d_priority,
	locateresult* d_pointlocation,
	trihandle* d_searchsh,
	tethandle* d_searchtet,
	int* d_insertidxlist,
	int* d_threadmarker,
	int* d_threadlist,
	REAL* d_steinerptlist,
	int numofsplittablesubsegs,
	int numofthreads
);

// Split encroached segment
__device__ int checkseg4split(
	trihandle* chkseg, 
	int& encpt, 
	REAL* pointlist, 
	int* seglist,
	tethandle* seg2tetlist,
	int* tetlist,
	tethandle* neighborlist);

__device__ int checkseg4encroach(
	REAL *pa, REAL* pb, REAL* checkpt
);

__global__ void kernelMarkAllEncsegs(
	REAL* d_pointlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	int* d_segencmarker,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int numofsubseg
);

__device__ void projectpoint2edge(
	REAL* p,
	REAL* e1,
	REAL* e2,
	REAL* prj
);

__global__ void kernelComputeSteinerPoint_Seg(
	int* d_threadlist,
	REAL* d_pointlist,
	trihandle* d_point2trilist,
	verttype* d_pointtypelist,
	int* d_seglist,
	int* d_seg2parentlist,
	int* d_segparentlist,
	int* d_segencmarker,
	int* d_encseglist,
	REAL* d_steinerptlist,
	int numofthreads
);

// Split encroached subface
__device__ int checkface4split(
	trihandle* chkfac,
	int& encpt,
	REAL* pointlist,
	int* trifacelist,
	tethandle* tri2tetlist,
	int* tetlist
);

__device__ int checkface4encroach(
	REAL *pa, REAL *pb, REAL *pc, REAL *checkpt
);

__global__ void kernelMarkAllEncsubfaces(
	REAL* d_pointlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	int* d_subfaceencmarker,
	int* d_tetlist,
	int numofsubface
);

__global__ void kernelComputeSteinerPoint_Subface(
	REAL* d_pointlist,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_encsubfacelist,
	REAL* d_steinerptlist,
	int numofencsubface
);

// Split bad tet
__device__ int checktet4split(
	tethandle* chktet,
	REAL* pointlist,
	int* tetlist,
	REAL minratio
);

__global__ void kernelMarkAllBadtets(
	REAL* d_pointlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL minratio,
	int numofbadtet
);

// Compact mesh
__global__ void kernelCompactSeg(
	int* d_seglist,
	int* d_sizes,
	int* d_indices,
	int* d_list,
	int numofthreads
);

__global__ void kernelCompactTriface(
	int* d_trifacelist,
	int* d_sizes,
	int* d_indices,
	int* d_list,
	int numofthreads
);

__global__ void kernelCompactTet_Phase1(
	int* d_trifacelist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int numofthreads
);

__global__ void kernelCompactTet_Phase2(
	int* d_tetlist,
	int* d_sizes,
	int* d_indices,
	int* d_list,
	int numofthreads
);

