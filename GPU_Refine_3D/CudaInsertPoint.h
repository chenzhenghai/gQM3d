#pragma once
#include "CudaThrust.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
//How to use:
//gpuErrchk(cudaPeekAtLastError());
//gpuErrchk(cudaDeviceSynchronize());

// 1: It is still possible to insert some points for current id list
// 0: It is impossible to insert any points for current id list if the mesh doesn't change
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
	IntD& t_threadmarker, // indicate insertion type: -1 failed, 0 splitsubseg, 1 splitsubface, 2 splittet
	int numofinsertpt,
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
);

// 1: It is still possible to insert some points for current id list
// 0: It is impossible to insert any points for current id list if the mesh doesn't change
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
);