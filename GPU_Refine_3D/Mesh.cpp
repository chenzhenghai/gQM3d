// This file is adapted from TetGen

#include "Mesh.h"
#include "MeshPredicates.h"
#include <stdio.h>
#include <assert.h>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Variables			                                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Helpers */

unsigned long randomseed;                    // Current random number seed.

/* Initialize fast lookup tables for mesh maniplulation primitives. */

int bondtbl[12][12] = { { 0, }, };
int fsymtbl[12][12] = { { 0, }, };
int enexttbl[12] = { 0, };
int eprevtbl[12] = { 0, };
int enextesymtbl[12] = { 0, };
int eprevesymtbl[12] = { 0, };
int eorgoppotbl[12] = { 0, };
int edestoppotbl[12] = { 0, };
int facepivot1[12] = { 0, };
int facepivot2[12][12] = { { 0, }, };
int tsbondtbl[12][6] = { { 0, }, };
int stbondtbl[12][6] = { { 0, }, };
int tspivottbl[12][6] = { { 0, }, };
int stpivottbl[12][6] = { { 0, }, };


// Table 'esymtbl' takes an directed edge (version) as input, returns the
//   inversed edge (version) of it.

int esymtbl[12] = { 9, 6, 11, 4, 3, 7, 1, 5, 10, 0, 8, 2 };

// The following four tables give the 12 permutations of the set {0,1,2,3}.

int orgpivot[12] = { 3, 3, 1, 1, 2, 0, 0, 2, 1, 2, 3, 0 };
int destpivot[12] = { 2, 0, 0, 2, 1, 2, 3, 0, 3, 3, 1, 1 };
int apexpivot[12] = { 1, 2, 3, 0, 3, 3, 1, 1, 2, 0, 0, 2 };
int oppopivot[12] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };

// The twelve versions correspond to six undirected edges. The following two
//   tables map a version to an undirected edge and vice versa.

int ver2edge[12] = { 0, 1, 2, 3, 3, 5, 1, 5, 4, 0, 4, 2 };
int edge2ver[6] = { 0, 1, 2, 3, 8, 5 };

// Table 'snextpivot' takes an edge version as input, returns the next edge
//   version in the same edge ring.

int snextpivot[6] = { 2, 5, 4, 1, 0, 3 };

// The following three tables give the 6 permutations of the set {0,1,2}.
//   An offset 3 is added to each element for a direct access of the points
//   in the triangle data structure.

int sorgpivot[6] = { 0, 1, 1, 2, 2, 0 };
int sdestpivot[6] = { 1, 0, 2, 1, 0, 2 };
int sapexpivot[6] = { 2, 2, 0, 0, 1, 1 };

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Helpers																	 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

unsigned long randomnation(unsigned int choices)
{
	unsigned long newrandom;

	if (choices >= 714025l) {
		newrandom = (randomseed * 1366l + 150889l) % 714025l;
		randomseed = (newrandom * 1366l + 150889l) % 714025l;
		newrandom = newrandom * (choices / 714025l) + randomseed;
		if (newrandom >= choices) {
			return newrandom - choices;
		}
		else {
			return newrandom;
		}
	}
	else {
		randomseed = (randomseed * 1366l + 150889l) % 714025l;
		return randomseed % choices;
	}
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Mesh manipulation primitives                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Initialize tables */
void inittables()
{
	int i, j;

	// i = t1.ver; j = t2.ver;
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			bondtbl[i][j] = (j & 3) + (((i & 12) + (j & 12)) % 12);
		}
	}

	// i = t1.ver; j = t2.ver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			fsymtbl[i][j] = (j + 12 - (i & 12)) % 12;
		}
	}

	for (i = 0; i < 12; i++) {
		facepivot1[i] = (esymtbl[i] & 3);
	}

	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			facepivot2[i][j] = fsymtbl[esymtbl[i]][j];
		}
	}

	for (i = 0; i < 12; i++) {
		enexttbl[i] = (i + 4) % 12;
		eprevtbl[i] = (i + 8) % 12;
	}

	for (i = 0; i < 12; i++) {
		enextesymtbl[i] = esymtbl[enexttbl[i]];
		eprevesymtbl[i] = esymtbl[eprevtbl[i]];
	}

	for (i = 0; i < 12; i++) {
		eorgoppotbl[i] = eprevtbl[esymtbl[enexttbl[i]]];
		edestoppotbl[i] = enexttbl[esymtbl[eprevtbl[i]]];
	}

	int soffset, toffset;

	// i = t.ver, j = s.shver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 6; j++) {
			if ((j & 1) == 0) {
				soffset = (6 - ((i & 12) >> 1)) % 6;
				toffset = (12 - ((j & 6) << 1)) % 12;
			}
			else {
				soffset = (i & 12) >> 1;
				toffset = (j & 6) << 1;
			}
			tsbondtbl[i][j] = (j & 1) + (((j & 6) + soffset) % 6);
			stbondtbl[i][j] = (i & 3) + (((i & 12) + toffset) % 12);
		}
	}


	// i = t.ver, j = s.shver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 6; j++) {
			if ((j & 1) == 0) {
				soffset = (i & 12) >> 1;
				toffset = (j & 6) << 1;
			}
			else {
				soffset = (6 - ((i & 12) >> 1)) % 6;
				toffset = (12 - ((j & 6) << 1)) % 12;
			}
			tspivottbl[i][j] = (j & 1) + (((j & 6) + soffset) % 6);
			stpivottbl[i][j] = (i & 3) + (((i & 12) + toffset) % 12);
		}
	}

}

/* Primitives for points */

// Convert point index to pointer to pointlist
double* id2pointlist(int index, double* pointlist)
{
	return (pointlist + 3 * index);
}

/* Primitives for tetrahedron */

// The following primtives get or set the origin, destination, face apex,
//   or face opposite of an ordered tetrahedron.

int org(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + orgpivot[t.ver]];
}

int dest(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + destpivot[t.ver]];
}

int apex(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + apexpivot[t.ver]];
}

int oppo(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + oppopivot[t.ver]];
}

// bond()  connects two tetrahedra together. (t1,v1) and (t2,v2) must 
//   refer to the same face and the same edge. 

void bond(tethandle t1, tethandle t2, tethandle* neighborlist)
{
	neighborlist[4 * t1.id + (t1.ver & 3)] = tethandle(t2.id, bondtbl[t1.ver][t2.ver]);
	neighborlist[4 * t2.id + (t2.ver & 3)] = tethandle(t1.id, bondtbl[t2.ver][t1.ver]);
}

// dissolve()  a bond (from one side).

void dissolve(tethandle t, tethandle* neighborlist)
{
	neighborlist[4 * t.id + (t.ver & 3)] = tethandle(-1, 11); // empty handle
}

// esym()  finds the reversed edge.  It is in the other face of the
//   same tetrahedron.

void esym(tethandle& t1, tethandle& t2)
{
	(t2).id = (t1).id;
	(t2).ver = esymtbl[(t1).ver];
}
void esymself(tethandle& t)
{
	(t).ver = esymtbl[(t).ver];
}

// enext()  finds the next edge (counterclockwise) in the same face.

void enext(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = enexttbl[t1.ver];
}
void enextself(tethandle& t)
{
	t.ver = enexttbl[t.ver];
}

// eprev()   finds the next edge (clockwise) in the same face.

void eprev(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = eprevtbl[t1.ver];
}
void eprevself(tethandle& t)
{
	t.ver = eprevtbl[t.ver];
}

// enextesym()  finds the reversed edge of the next edge. It is in the other
//   face of the same tetrahedron. It is the combination esym() * enext(). 

void enextesym(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = enextesymtbl[t1.ver];
}

void enextesymself(tethandle& t) {
	t.ver = enextesymtbl[t.ver];
}

// eprevesym()  finds the reversed edge of the previous edge.

void eprevesym(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = eprevesymtbl[t1.ver];
}

void eprevesymself(tethandle& t) {
	t.ver = eprevesymtbl[t.ver];
}

// eorgoppo()    Finds the opposite face of the origin of the current edge.
//               Return the opposite edge of the current edge.

void eorgoppo(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = eorgoppotbl[t1.ver];
}

void eorgoppoself(tethandle& t) {
	t.ver = eorgoppotbl[t.ver];
}

// edestoppo()    Finds the opposite face of the destination of the current 
//                edge. Return the opposite edge of the current edge.

void edestoppo(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = edestoppotbl[t1.ver];
}

void edestoppoself(tethandle& t) {
	t.ver = edestoppotbl[t.ver];
}

// fsym()  finds the adjacent tetrahedron at the same face and the same edge.

void fsym(tethandle& t1, tethandle& t2, tethandle* neighborlist)
{
	t2 = neighborlist[4 * t1.id + (t1.ver & 3)];
	t2.ver = fsymtbl[t1.ver][t2.ver];
}

void fsymself(tethandle& t, tethandle* neighborlist)
{
	char t1ver = t.ver;
	t = neighborlist[4 * t.id + (t.ver & 3)];
	t.ver = fsymtbl[t1ver][t.ver];
}

// fnext()  finds the next face while rotating about an edge according to
//   a right-hand rule. The face is in the adjacent tetrahedron.  It is
//   the combination: fsym() * esym().

void fnext(tethandle& t1, tethandle& t2, tethandle* neighborlist) 
{
	t2 = neighborlist[4 * t1.id + facepivot1[t1.ver]];
	t2.ver = facepivot2[t1.ver][t2.ver];
}

void fnextself(tethandle& t, tethandle* neighborlist)
{
	char t1ver = t.ver;
	t = neighborlist[4 * t.id + facepivot1[t.ver]];
	t.ver = facepivot2[t1ver][t.ver];
}

// ishulltet()  tests if t is a hull tetrahedron.

bool ishulltet(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + 3] == -1;
}

// isdeadtet()  tests if t is a tetrahedron is dead.

bool isdeadtet(tethandle t)
{
	return (t.id == -1);
}

/* Primitives for subfaces and subsegments. */

// spivot() finds the adjacent subface (s2) for a given subface (s1).
//   s1 and s2 share at the same edge.

void spivot(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	s2 = tri2trilist[3 * s1.id + (s1.shver >> 1)];
}

void spivotself(trihandle& s, trihandle* tri2trilist)
{
	s = tri2trilist[3 * s.id + (s.shver >> 1)];
}

// sbond() bonds two subfaces (s1) and (s2) together. s1 and s2 must refer
//   to the same edge. No requirement is needed on their orientations.

void sbond(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	tri2trilist[3 * s1.id + (s1.shver >> 1)] = s2;
	tri2trilist[3 * s2.id + (s2.shver >> 1)] = s1;
}

// sbond1() bonds s1 <== s2, i.e., after bonding, s1 is pointing to s2,
//   but s2 is not pointing to s1.  s1 and s2 must refer to the same edge.
//   No requirement is needed on their orientations.

void sbond1(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	tri2trilist[3 * s1.id + (s1.shver >> 1)] = s2;
}

// These primitives determine or set the origin, destination, or apex
//   of a subface with respect to the edge version.

int sorg(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + sorgpivot[s.shver]];
}

int sdest(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + sdestpivot[s.shver]];
}

int sapex(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + sapexpivot[s.shver]];
}

// sesym()  reserves the direction of the lead edge.

void sesym(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = (s1.shver ^ 1);  // Inverse the last bit.
}

void sesymself(trihandle& s)
{
	s.shver ^= 1;
}

// senext()  finds the next edge (counterclockwise) in the same orientation
//   of this face.

void senext(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = snextpivot[s1.shver];
}

void senextself(trihandle& s)
{
	s.shver = snextpivot[s.shver];
}

void senext2(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = snextpivot[snextpivot[s1.shver]];
}

void senext2self(trihandle& s)
{
	s.shver = snextpivot[snextpivot[s.shver]];
}


/* Primitives for interacting tetrahedra and subfaces. */

// tsbond() bond a tetrahedron (t) and a subface (s) together.
// Note that t and s must be the same face and the same edge. Moreover,
//   t and s have the same orientation. 
// Since the edge number in t and in s can be any number in {0,1,2}. We bond
//   the edge in s which corresponds to t's 0th edge, and vice versa.

void tsbond(tethandle& t, trihandle& s, trihandle* tet2trilist, tethandle* tri2tetlist)
{
	// Bond t <== s.
	tet2trilist[4 * t.id + (t.ver & 3)] = trihandle(s.id, tsbondtbl[t.ver][s.shver]);
	// Bond s <== t.
	tri2tetlist[2 * s.id + (s.shver & 1)] = tethandle(t.id, stbondtbl[t.ver][s.shver]);
}

// tspivot() finds a subface (s) abutting on the given tetrahdera (t).
//   Return s.id = -1 if there is no subface at t. Otherwise, return
//   the subface s, and s and t must be at the same edge wth the same
//   orientation.
void tspivot(tethandle& t, trihandle& s, trihandle* tet2trilist)
{
	// Get the attached subface s.
	s = tet2trilist[4 * t.id + (t.ver & 3)];
	if (s.id == -1)
		return;
	(s).shver = tspivottbl[t.ver][s.shver];
}

// stpivot() finds a tetrahedron (t) abutting a given subface (s).
//   Return the t (if it exists) with the same edge and the same
//   orientation of s.
void stpivot(trihandle& s, tethandle& t, tethandle* tri2tetlist)
{
	t = tri2tetlist[2 * s.id + (s.shver & 1)];
	if (t.id == -1) {
		return;
	}
	(t).ver = stpivottbl[t.ver][s.shver];
}

/* Primitives for interacting between tetrahedra and segments */

void tssbond1(tethandle& t, trihandle& seg, trihandle* tet2seglist)
{
	tet2seglist[6 * t.id + ver2edge[t.ver]] = seg;
}

void sstbond1(trihandle& s, tethandle& t, tethandle* seg2tetlist)
{
	seg2tetlist[s.id + 0] = t;
}

/* Primitives for interacting between subfaces and segments */

void ssbond(trihandle& s, trihandle& edge, trihandle* tri2seglist, trihandle* seg2trilist)
{
	tri2seglist[3 * s.id + (s.shver >> 1)] = edge;
	seg2trilist[3 * edge.id + 0] = s;
}

void ssbond1(trihandle& s, trihandle& edge, trihandle* tri2seglist)
{
	tri2seglist[3 * s.id + (s.shver >> 1)] = edge;
}

void sspivot(trihandle& s, trihandle& edge, trihandle* tri2seglist)
{
	edge = tri2seglist[3 * s.id + (s.shver >> 1)];
}

bool isshsubseg(trihandle& s, trihandle* tri2seglist)
{
	return (tri2seglist[3 * s.id + (s.shver >> 1)].id != -1);
}

/* Advanced primitives. */

void point2tetorg(int pa, tethandle& searchtet, tethandle* point2tet, int* tetlist)
{
	searchtet = point2tet[pa];
	if (tetlist[4*searchtet.id + 0] == pa) {
		searchtet.ver = 11;
	}
	else if (tetlist[4*searchtet.id + 1] == pa) {
		searchtet.ver = 3;
	}
	else if (tetlist[4*searchtet.id + 2] == pa) {
		searchtet.ver = 7;
	}
	else {
		assert(tetlist[4*searchtet.id + 3] == pa); // SELF_CHECK
		searchtet.ver = 0;
	}
}

// distance() computes the Euclidean distance between two points.
REAL pointdistance(REAL* p1, REAL* p2)
{
	return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
		(p2[1] - p1[1]) * (p2[1] - p1[1]) +
		(p2[2] - p1[2]) * (p2[2] - p1[2]));
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Linear algebra operators.                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


// dot() returns the dot product: v1 dot v2.
REAL meshdot(REAL* v1, REAL* v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// cross() computes the cross product: n = v1 cross v2.
void meshcross(REAL* v1, REAL* v2, REAL* n)
{
	n[0] = v1[1] * v2[2] - v2[1] * v1[2];
	n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
	n[2] = v1[0] * v2[1] - v2[0] * v1[1];
}

// facenormal() Calculate the normal of the face. 
void meshfacenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
	REAL* lav)
{
	REAL v1[3], v2[3], v3[3], *pv1, *pv2;
	REAL L1, L2, L3;

	v1[0] = pb[0] - pa[0];  // edge vector v1: a->b
	v1[1] = pb[1] - pa[1];
	v1[2] = pb[2] - pa[2];
	v2[0] = pa[0] - pc[0];  // edge vector v2: c->a
	v2[1] = pa[1] - pc[1];
	v2[2] = pa[2] - pc[2];

	// Default, normal is calculated by: v1 x (-v2) (see Fig. fnormal).
	if (pivot > 0) {
		// Choose edge vectors by Burdakov's algorithm.
		v3[0] = pc[0] - pb[0];  // edge vector v3: b->c
		v3[1] = pc[1] - pb[1];
		v3[2] = pc[2] - pb[2];
		L1 = meshdot(v1, v1);
		L2 = meshdot(v2, v2);
		L3 = meshdot(v3, v3);
		// Sort the three edge lengths.
		if (L1 < L2) {
			if (L2 < L3) {
				pv1 = v1; pv2 = v2; // n = v1 x (-v2).
			}
			else {
				pv1 = v3; pv2 = v1; // n = v3 x (-v1).
			}
		}
		else {
			if (L1 < L3) {
				pv1 = v1; pv2 = v2; // n = v1 x (-v2).
			}
			else {
				pv1 = v2; pv2 = v3; // n = v2 x (-v3).
			}
		}
		if (lav) {
			// return the average edge length.
			*lav = (sqrt(L1) + sqrt(L2) + sqrt(L3)) / 3.0;
		}
	}
	else {
		pv1 = v1; pv2 = v2; // n = v1 x (-v2).
	}

	// Calculate the face normal.
	meshcross(pv1, pv2, n);
	// Inverse the direction;
	n[0] = -n[0];
	n[1] = -n[1];
	n[2] = -n[2];
}


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// finddirection()    Find the tet on the path from one point to another.    //
//                                                                           //
// The path starts from 'searchtet''s origin and ends at 'endpt'. On finish, //
// 'searchtet' contains a tet on the path, its origin does not change.       //
//                                                                           //
// The return value indicates one of the following cases (let 'searchtet' be //
// abcd, a is the origin of the path):                                       //
//   - ACROSSVERT, edge ab is collinear with the path;                       //
//   - ACROSSEDGE, edge bc intersects with the path;                         //
//   - ACROSSFACE, face bcd intersects with the path.                        //
//                                                                           //
// WARNING: This routine is designed for convex triangulations, and will not //
// generally work after the holes and concavities have been carved.          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

enum interresult finddirection(tethandle* searchtet, int endpt, double* pointlist, int* tetlist, tethandle* neighborlist)
{
	tethandle neightet;
	int pa, pb, pc, pd;
	enum { HMOVE, RMOVE, LMOVE } nextmove;
	REAL hori, rori, lori;
	int t1ver;
	int s;

	// The origin is fixed.
	pa = org(*searchtet,tetlist);
	if (tetlist[4 * searchtet->id + 3] == -1)
	{
		// A hull tet. Choose the neighbor of its base face.
		*searchtet = neighborlist[4 * searchtet->id + 3];
		// Reset the origin to be pa.
		if (tetlist[4 * searchtet->id + 0] == pa) 
		{
			searchtet->ver = 11;
		}
		else if (tetlist[4 * searchtet->id + 1] == pa) 
		{
			searchtet->ver = 3;
		}
		else if (tetlist[4 * searchtet->id + 2] == pa) {
			searchtet->ver = 7;
		}
		else {
			assert(tetlist[4 * searchtet->id + 3] == pa);
			searchtet->ver = 0;
		}
	}

	pb = dest(*searchtet, tetlist);
	// Check whether the destination or apex is 'endpt'.
	if (pb == endpt) {
		// pa->pb is the search edge.
		return ACROSSVERT;
	}

	pc = apex(*searchtet, tetlist);
	if (pc == endpt) {
		// pa->pc is the search edge.
		eprevesymself(*searchtet);
		return ACROSSVERT;
	}

	double *p[5];

	// Walk through tets around pa until the right one is found.
	while (1) {

		pd = oppo(*searchtet, tetlist);
		// Check whether the opposite vertex is 'endpt'.
		if (pd == endpt) {
			// pa->pd is the search edge.
			esymself(*searchtet);
			enextself(*searchtet);
			return ACROSSVERT;
		}
		// Check if we have entered outside of the domain.
		if (pd == -1) {
			// This is possible when the mesh is non-convex.
			return ACROSSSUB; // Hit a boundary.
		}

		// Now assume that the base face abc coincides with the horizon plane,
		//   and d lies above the horizon.  The search point 'endpt' may lie
		//   above or below the horizon.  We test the orientations of 'endpt'
		//   with respect to three planes: abc (horizon), bad (right plane),
		//   and acd (left plane).
		p[0] = id2pointlist(pa, pointlist);
		p[1] = id2pointlist(pb, pointlist);
		p[2] = id2pointlist(pc, pointlist);
		p[3] = id2pointlist(pd, pointlist);
		p[4] = id2pointlist(endpt, pointlist);
		
		hori = orient3d(p[0], p[1], p[2], p[4]);
		rori = orient3d(p[1], p[0], p[3], p[4]);
		lori = orient3d(p[0], p[2], p[3], p[4]);

		// Now decide the tet to move.  It is possible there are more than one
		//   tets are viable moves. Is so, randomly choose one. 
		if (hori > 0) {
			if (rori > 0) {
				if (lori > 0) {
					// Any of the three neighbors is a viable move.
					s = randomnation(3);
					if (s == 0) {
						nextmove = HMOVE;
					}
					else if (s == 1) {
						nextmove = RMOVE;
					}
					else {
						nextmove = LMOVE;
					}
				}
				else {
					// Two tets, below horizon and below right, are viable.
					//s = randomnation(2); 
					if (randomnation(2)) {
						nextmove = HMOVE;
					}
					else {
						nextmove = RMOVE;
					}
				}
			}
			else {
				if (lori > 0) {
					// Two tets, below horizon and below left, are viable.
					//s = randomnation(2); 
					if (randomnation(2)) {
						nextmove = HMOVE;
					}
					else {
						nextmove = LMOVE;
					}
				}
				else {
					// The tet below horizon is chosen.
					nextmove = HMOVE;
				}
			}
		}
		else {
			if (rori > 0) {
				if (lori > 0) {
					// Two tets, below right and below left, are viable.
					//s = randomnation(2); 
					if (randomnation(2)) {
						nextmove = RMOVE;
					}
					else {
						nextmove = LMOVE;
					}
				}
				else {
					// The tet below right is chosen.
					nextmove = RMOVE;
				}
			}
			else {
				if (lori > 0) {
					// The tet below left is chosen.
					nextmove = LMOVE;
				}
				else {
					// 'endpt' lies either on the plane(s) or across face bcd.
					if (hori == 0) {
						if (rori == 0) {
							// pa->'endpt' is COLLINEAR with pa->pb.
							return ACROSSVERT;
						}
						if (lori == 0) {
							// pa->'endpt' is COLLINEAR with pa->pc.
							eprevesymself(*searchtet); // // [a,c,d]
							return ACROSSVERT;
						}
						// pa->'endpt' crosses the edge pb->pc.
						return ACROSSEDGE;
					}
					if (rori == 0) {
						if (lori == 0) {
							// pa->'endpt' is COLLINEAR with pa->pd.
							esymself(*searchtet); // face bad.
							enextself(*searchtet); // face [a,d,b]
							return ACROSSVERT;
						}
						// pa->'endpt' crosses the edge pb->pd.
						esymself(*searchtet); // face bad.
						enextself(*searchtet); // face adb
						return ACROSSEDGE;
					}
					if (lori == 0) {
						// pa->'endpt' crosses the edge pc->pd.
						eprevesymself(*searchtet); // [a,c,d]
						return ACROSSEDGE;
					}
					// pa->'endpt' crosses the face bcd.
					return ACROSSFACE;
				}
			}
		}

		// Move to the next tet, fix pa as its origin.
		if (nextmove == RMOVE) {
			fnextself(*searchtet, neighborlist);
		}
		else if (nextmove == LMOVE) {
			eprevself(*searchtet);
			fnextself(*searchtet, neighborlist);
			enextself(*searchtet);
		}
		else { // HMOVE
			fsymself(*searchtet, neighborlist);
			enextself(*searchtet);
		}
		assert(org(*searchtet, tetlist) == pa);
		pb = dest(*searchtet, tetlist);
		pc = apex(*searchtet, tetlist);

	} // while (1)
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// getedge()    Get a tetrahedron having the two endpoints.                  //
//                                                                           //
// The method here is to search the second vertex in the link faces of the   //
// first vertex. The global array 'cavetetlist' is re-used for searching.    //
//                                                                           //
// This function is used for the case when the mesh is non-convex. Otherwise,//
// the function finddirection() should be faster than this.                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

int getedge(int e1, int e2, tethandle *tedge, tethandle* point2tet, double* pointlist, int* tetlist, tethandle* neighborlist, int* markerlist)
{
	tethandle searchtet, neightet, parytet;
	int pt;
	int done;
	int i, j;

	// Quickly check if 'tedge' is just this edge.
	if (!isdeadtet(*tedge)) {
		if (org(*tedge, tetlist) == e1) {
			if (dest(*tedge, tetlist) == e2) {
				return 1;
			}
		}
		else if (org(*tedge, tetlist) == e2) {
			if (dest(*tedge, tetlist) == e1) {
				esymself(*tedge);
				return 1;
			}
		}
	}

	// Search for the edge [e1, e2].
	point2tetorg(e1, *tedge, point2tet, tetlist);
	finddirection(tedge, e2, pointlist, tetlist, neighborlist);
	if (dest(*tedge, tetlist) == e2)
	{
		return 1;
	}
	else
	{
		// Search for the edge [e2, e1].
		point2tetorg(e2, *tedge, point2tet, tetlist);
		finddirection(tedge, e1, pointlist, tetlist, neighborlist);
		if (dest(*tedge, tetlist) == e1) {
			esymself(*tedge);
			return 1;
		}
	}

	// Go to the link face of e1.
	point2tetorg(e1, searchtet, point2tet, tetlist);
	enextesymself(searchtet);

	std::vector<tethandle> recordtetlist; // recorded tet list

	// Search e2.
	for (i = 0; i < 3; i++) {
		pt = apex(searchtet, tetlist);
		if (pt == e2) {
			// Found. 'searchtet' is [#,#,e2,e1].
			eorgoppo(searchtet, *tedge); // [e1,e2,#,#].
			return 1;
		}
		enextself(searchtet);
	}

	// Get the adjacent link face at 'searchtet'.
	fnext(searchtet, neightet, neighborlist);
	esymself(neightet);
	// assert(oppo(neightet) == e1);
	pt = apex(neightet, tetlist);
	if (pt == e2) {
		// Found. 'neightet' is [#,#,e2,e1].
		eorgoppo(neightet, *tedge); // [e1,e2,#,#].
		return 1;
	}

	// Continue searching in the link face of e1.
	markerlist[searchtet.id] = 1; // initial value of markerlist must be 0
	recordtetlist.push_back(searchtet);
	markerlist[neightet.id] = 1;
	recordtetlist.push_back(neightet);

	done = 0;

	for (i = 0; (i < recordtetlist.size()) && !done; i++) {
		parytet = recordtetlist[i];
		searchtet = parytet;
		for (j = 0; (j < 2) && !done; j++) {
			enextself(searchtet);
			fnext(searchtet, neightet, neighborlist);
			if (!markerlist[neightet.id]) {
				esymself(neightet);
				pt = apex(neightet, tetlist);
				if (pt == e2) {
					// Found. 'neightet' is [#,#,e2,e1].
					eorgoppo(neightet, *tedge);
					done = 1;
				}
				else {
					markerlist[neightet.id] = 1;
					recordtetlist.push_back(neightet);
				}
			}
		} // j
	} // i 

	// Uninfect the list of visited tets.
	for (i = 0; i < recordtetlist.size(); i++) {
		parytet = recordtetlist[i];
		markerlist[parytet.id] = 0;
	}

	return done;
}

bool meshludecmp(REAL lu[4][4], int n, int* ps, REAL* d, int N)
{
	REAL scales[4];
	REAL pivot, biggest, mult, tempf;
	int pivotindex = 0;
	int i, j, k;

	*d = 1.0;                                      // No row interchanges yet.

	for (i = N; i < n + N; i++) {                             // For each row.
															  // Find the largest element in each row for row equilibration
		biggest = 0.0;
		for (j = N; j < n + N; j++)
			if (biggest < (tempf = fabs(lu[i][j])))
				biggest = tempf;
		if (biggest != 0.0)
			scales[i] = 1.0 / biggest;
		else {
			scales[i] = 0.0;
			return false;                            // Zero row: singular matrix.
		}
		ps[i] = i;                                 // Initialize pivot sequence.
	}

	for (k = N; k < n + N - 1; k++) {                      // For each column.
														   // Find the largest element in each column to pivot around.
		biggest = 0.0;
		for (i = k; i < n + N; i++) {
			if (biggest < (tempf = fabs(lu[ps[i]][k]) * scales[ps[i]])) {
				biggest = tempf;
				pivotindex = i;
			}
		}
		if (biggest == 0.0) {
			return false;                         // Zero column: singular matrix.
		}
		if (pivotindex != k) {                         // Update pivot sequence.
			j = ps[k];
			ps[k] = ps[pivotindex];
			ps[pivotindex] = j;
			*d = -(*d);                          // ...and change the parity of d.
		}

		// Pivot, eliminating an extra variable  each time
		pivot = lu[ps[k]][k];
		for (i = k + 1; i < n + N; i++) {
			lu[ps[i]][k] = mult = lu[ps[i]][k] / pivot;
			if (mult != 0.0) {
				for (j = k + 1; j < n + N; j++)
					lu[ps[i]][j] -= mult * lu[ps[k]][j];
			}
		}
	}

	// (lu[ps[n + N - 1]][n + N - 1] == 0.0) ==> A is singular.
	return lu[ps[n + N - 1]][n + N - 1] != 0.0;
}

void meshlusolve(REAL lu[4][4], int n, int* ps, REAL* b, int N)
{
	int i, j;
	REAL X[4], dot;

	for (i = N; i < n + N; i++) X[i] = 0.0;

	// Vector reduction using U triangular matrix.
	for (i = N; i < n + N; i++) {
		dot = 0.0;
		for (j = N; j < i + N; j++)
			dot += lu[ps[i]][j] * X[j];
		X[i] = b[ps[i]] - dot;
	}

	// Back substitution, in L triangular matrix.
	for (i = n + N - 1; i >= N; i--) {
		dot = 0.0;
		for (j = i + 1; j < n + N; j++)
			dot += lu[ps[i]][j] * X[j];
		X[i] = (X[i] - dot) / lu[ps[i]][i];
	}

	for (i = N; i < n + N; i++) b[i] = X[i];
}
