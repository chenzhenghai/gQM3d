#pragma once

#include "MeshStructure.h"

#define REAL double
#define EPSILON 1.0e-8
#define PI 3.14159265358979323846264338327950288419716939937510582

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Helpers																	 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

unsigned long randomnation(unsigned int choices);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Mesh manipulation primitives                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Init fast lookup tables */
void inittables();

/* Primitives for points */
// Convert point index to pointer to pointlist
double* id2pointlist(int index, double* pointlist);

/* Primitives for tetrahedron */

int org(tethandle t, int* tetlist);
int dest(tethandle t, int* tetlist);
int apex(tethandle t, int* tetlist);
int oppo(tethandle t, int* tetlist);

void bond(tethandle t1, tethandle t2, tethandle* neighborlist);
void dissolve(tethandle t, tethandle* neighborlist);

void esym(tethandle& t1, tethandle& t2);
void esymself(tethandle& t);
void enext(tethandle& t1, tethandle& t2);
void enextself(tethandle& t);
void eprev(tethandle& t1, tethandle& t2);
void eprevself(tethandle& t);
void enextesym(tethandle& t1, tethandle& t2);
void enextesymself(tethandle& t);
void eprevesym(tethandle& t1, tethandle& t2);
void eprevesymself(tethandle& t);
void eorgoppo(tethandle& t1, tethandle& t2);
void eorgoppoself(tethandle& t);
void edestoppo(tethandle& t1, tethandle& t2);
void edestoppoself(tethandle& t);

void fsym(tethandle& t1, tethandle& t2, tethandle* neighborlist);
void fsymself(tethandle& t, tethandle* neighborlist);
void fnext(tethandle& t1, tethandle& t2, tethandle* neigenhborlist);
void fnextself(tethandle& t, tethandle* neighborlist);

bool ishulltet(tethandle t, int* tetlist);
bool isdeadtet(tethandle t);

enum interresult finddirection(tethandle* searchtet, int endpt, double* pointlist, int* tetlist, tethandle* neighborlist);
int getedge(int e1, int e2, tethandle *tedge, tethandle* point2tet, double* pointlist, int* tetlist, tethandle* neighborlist, int* markerlist);

// Primitives for subfaces and subsegments.
void spivot(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
void spivotself(trihandle& s, trihandle* tri2trilist);
void sbond(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
void sbond1(trihandle& s1, trihandle& s2, trihandle* tri2trilist);
int sorg(trihandle& s, int* trilist);
int sdest(trihandle& s, int* trilist);
int sapex(trihandle& s, int* trilist);
void sesym(trihandle& s1, trihandle& s2);
void sesymself(trihandle& s);
void senext(trihandle& s1, trihandle& s2);
void senextself(trihandle& s);
void senext2(trihandle& s1, trihandle& s2);
void senext2self(trihandle& s);

// Primitives for interacting tetrahedra and subfaces.
void tsbond(tethandle& t, trihandle& s, trihandle* tet2trilist, tethandle* tri2tetlist);
void tspivot(tethandle& t, trihandle& s, trihandle* tet2trilist);
void stpivot(trihandle& s, tethandle& t, tethandle* tri2tetlist);

// Primitives for interacting tetrahedra and segments.
void tssbond1(tethandle& t, trihandle& seg, trihandle* tet2seglist);
void sstbond1(trihandle& s, tethandle& t, tethandle* seg2tetlist);

// Primitives for interacting subfaces and segments.
void ssbond(trihandle& s, trihandle& edge, trihandle* tri2seglist, trihandle* seg2trilist);
void ssbond1(trihandle& s, trihandle& edge, trihandle* tri2seglist);
void sspivot(trihandle& s, trihandle& edge, trihandle* tri2seglist);
bool isshsubseg(trihandle& s, trihandle* tri2seglist);

/* Advanced primitives. */
void point2tetorg(int pa, tethandle& searchtet, tethandle* point2tet, int* tetlist);

/* Geometric calculations (non-robust) */
REAL pointdistance(REAL* p1, REAL* p2);

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Linear algebra operators.                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// Linear algebra functions
REAL meshdot(REAL* v1, REAL* v2);
void meshcross(REAL* v1, REAL* v2, REAL* n);
void meshfacenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
	REAL* lav);

bool meshludecmp(REAL lu[4][4], int n, int* ps, REAL* d, int N);
void meshlusolve(REAL lu[4][4], int n, int* ps, REAL* b, int N);