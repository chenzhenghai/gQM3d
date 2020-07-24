#pragma once
#include "InputGenerator.h"

void generatePLCs();
void generateCDTs();
void refineMeshCPU();
void generatePLCs_without_acute_angles();
void generateCDTs_without_acute_angles();
void refineMeshCPU_without_acute_angles();

void saveGPUResult_without_acute_angles(
	int numofpoint, int numoftri, int numofedge,
	int seed, Distribution dist, double minarea, double radio,
	int outnumofpoint, int outnumoftrifaces, int outnumofedges,
	int outnumoftets, int outnumofbadtets, double* times, int mode
);

void experiment_statistic();