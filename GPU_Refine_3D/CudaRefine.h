#pragma once

#include <cuda_runtime.h>
#include <helper_timer.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "MeshStructure.h"

#define MAXINT 2147483647
#define PI 3.141592653589793238462643383279502884197169399375105820974944592308

void GPU_Refine_3D(
	MESHIO* input_gpu,
	MESHBH* input_behavior,
	int& out_numofpoint,
	double*& out_pointlist,
	int& out_numofedge,
	int*& out_edgelist,
	int& out_numoftriface,
	int*& out_trifacelist,
	int& out_numoftet,
	int*& out_tetlist
);