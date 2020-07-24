#pragma once

#include "CudaThrust.h"

void compactMesh(
	int& out_numofpoint,
	double*& out_pointlist,
	RealD& t_pointlist,
	int& out_numofedge,
	int*& out_edgelist,
	IntD& t_seglist,
	TriStatusD& t_segstatus,
	int& out_numoftriface,
	int*& out_trifacelist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	int& out_numoftet,
	int*& out_tetlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus
);