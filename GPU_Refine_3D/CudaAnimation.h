#pragma once
#include "MeshStructure.h"
#include "CudaThrust.h"

void outputStartingFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	IntD& t_threadlist,
	IntD& t_insertidxlist,
	RealD& t_insertptlist,
	TetHandleD& t_locatedtet,
	int iter_seg,
	int iter_subface,
	int iter_tet
);

void outputCavityFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_tetlist,
	UInt64D& t_tetmarker,
	IntD& t_threadmarker,
	TetHandleD& t_caveoldtetlist,
	IntD& t_caveoldtetnext,
	IntD& t_caveoldtethead,
	int iter_seg,
	int iter_subface,
	int iter_tet,
	int iter_expanding,
	int expandingsize
);

void outputCavityFrame(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	TetHandleD& t_tri2tetlist,
	IntD& t_tetlist,
	TetHandleD& t_neighborlist,
	UInt64D& t_tetmarker,
	IntD& t_threadmarker,
	TetHandleD& t_caveoldtetlist,
	IntD& t_caveoldtetnext,
	IntD& t_caveoldtethead,
	int iter_seg,
	int iter_subface,
	int iter_tet,
	int iter_expanding,
	int expandingsize
);

void outputTmpMesh(
	internalmesh* drawmesh,
	RealD& t_pointlist,
	PointTypeD& t_pointtypelist,
	IntD& t_seglist,
	TriStatusD& t_segstatus,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus,
	IntD& t_insertidxlist,
	RealD& t_insertptlist,
	IntD& t_threadlist,
	IntD& t_threadmarker,
	TetHandleD& t_cavebdrylist,
	IntD& t_cavebdrynext,
	IntD& t_cavebdryhead,
	int insertiontype
);
