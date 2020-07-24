#pragma once

// Adapted from tetgen, for input generator only

#define REAL double

typedef REAL *point;

// Labels that signify the result of triangle-triangle intersection test.
enum interresult {
	DISJOINT, INTERSECT, SHAREVERT, SHAREEDGE, SHAREFACE,
	TOUCHEDGE, TOUCHFACE, ACROSSVERT, ACROSSEDGE, ACROSSFACE,
	COLLISIONFACE, ACROSSSEG, ACROSSSUB
};

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Robust Geometric predicates                                               //
//                                                                           //
// Geometric predicates are simple tests of spatial relations of a set of d- //
// dimensional points, such as the orientation test and the point-in-sphere  //
// test. Each of these tests is performed by evaluating the sign of a deter- //
// minant of a matrix whose entries are the coordinates of these points.  If //
// the computation is performed by using the floating-point numbers, e.g.,   //
// the single or double precision numbers in C/C++, roundoff error may cause //
// an incorrect result. This may either lead to a wrong result or eventually //
// lead to a failure of the program.  Computing the predicates exactly will  //
// avoid the error and make the program robust.                              //
//                                                                           //
// The following routines are the robust geometric predicates for 3D orient- //
// ation test and point-in-sphere test.  They were implemented by Shewchuk.  //
// The source code are generously provided by him in the public domain,      //
// http://www.cs.cmu.edu/~quake/robust.html. predicates.cxx is a C++ version //
// of the original C code.                                                   //
//                                                                           //
// The original predicates of Shewchuk only use "dynamic filters", i.e., it  //
// computes the error at run time step by step. TetGen first adds a "static  //
// filter" in each predicate. It estimates the maximal possible error in all //
// cases.  So it can safely and quickly answer many easy cases.              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

REAL orient2d(REAL *pa, REAL *pb, REAL *pc);
REAL orient3d(REAL *pa, REAL *pb, REAL *pc, REAL *pd);

// Linear algebra functions
inline REAL dot(REAL* v1, REAL* v2);
inline void cross(REAL* v1, REAL* v2, REAL* n);

// Geometric calculations (non-robust)
inline REAL distance(REAL* p1, REAL* p2);

// dot() returns the dot product: v1 dot v2.
inline REAL dot(REAL* v1, REAL* v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// cross() computes the cross product: n = v1 cross v2.
inline void cross(REAL* v1, REAL* v2, REAL* n)
{
	n[0] = v1[1] * v2[2] - v2[1] * v1[2];
	n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
	n[2] = v1[0] * v2[1] - v2[0] * v1[1];
}

// distance() computes the Euclidean distance between two points.
inline REAL distance(REAL* p1, REAL* p2)
{
	return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
		(p2[1] - p1[1]) * (p2[1] - p1[1]) +
		(p2[2] - p1[2]) * (p2[2] - p1[2]));
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// tri_tri_inter()    Test whether two triangle (abc) and (opq) are          //
//                    intersecting or not.                                   //
//                                                                           //
// Return 0 if they are disjoint. Otherwise, return 1. 'type' returns one of //
// the four cases: SHAREVERTEX, SHAREEDGE, SHAREFACE, and INTERSECT.         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

int tri_tri_inter(REAL* A, REAL* B, REAL* C, REAL* O, REAL* P, REAL* Q);
int tri_edge_test(point A, point B, point C, point P, point Q,
	point R, int level, int *types, int *pos);
bool rect_rect_inter(REAL* A, REAL* B, REAL* C, REAL* D,
	REAL* O, REAL* P, REAL* Q, REAL* R);