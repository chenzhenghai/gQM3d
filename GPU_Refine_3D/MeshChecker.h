#pragma once

int isBadTet(
	REAL *pa, REAL *pb, REAL *pc, REAL *pd,
	REAL minratio
);

int isBadTet(
	REAL *pa, REAL *pb, REAL *pc, REAL *pd,
	REAL minratio, REAL* ccent
);

REAL calRatio(
	REAL *pa, REAL *pb, REAL *pc, REAL *pd
);

void calDihedral(
	REAL **p,
	REAL *alldihed
);

int countBadTets(
	double* pointlist,
	int* tetlist,
	int numoftet,
	double minratio
);

void checkCDTMesh(
	int numofpoint,
	double* pointlist,
	int numofedge,
	int* edgelist,
	int numoftriface,
	int* trifacelist,
	int numoftet,
	int* tetlist
);