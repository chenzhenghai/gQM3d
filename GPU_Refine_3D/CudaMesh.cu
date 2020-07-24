// This file is adapted from TetGen

#include "CudaMesh.h"
#include "CudaPredicates.h"
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <assert.h>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Variables			                                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Kernel constants */

__constant__ REAL raw_kernelconstants[2];

REAL host_kernelconstants[2];

/* Helpers */
__device__ uint64 cudamesh_encodeUInt64Priority(int priority, int index)
{
	return (((uint64)priority) << 32) + index;
}

__device__ int cudamesh_getUInt64PriorityIndex(uint64 priority)
{
	return (priority & 0xFFFFFFFF);
}

__device__ int cudamesh_getUInt64Priority(uint64 priority)
{
	return (priority >> 32);
}

__device__ bool cudamesh_isNearZero(double val)
{
	if (val > -EPSILON && val < EPSILON)
		return true;
	else
		return false;
}

__device__ bool cudamesh_isInvalid(double val)
{
	if (val > 10000000 || val < -10000000)
		return true;
	else
		return false;
}

/* Initialize fast lookup tables for mesh maniplulation primitives. */

__constant__ int raw_bondtbl[144];
__constant__ int raw_fsymtbl[144];
__constant__ int raw_enexttbl[12];
__constant__ int raw_eprevtbl[12];
__constant__ int raw_enextesymtbl[12];
__constant__ int raw_eprevesymtbl[12];
__constant__ int raw_eorgoppotbl[12];
__constant__ int raw_edestoppotbl[12];
__constant__ int raw_facepivot1[12];
__constant__ int raw_facepivot2[144];
__constant__ int raw_tsbondtbl[72];
__constant__ int raw_stbondtbl[72];
__constant__ int raw_tspivottbl[72];
__constant__ int raw_stpivottbl[72];

int host_bondtbl[144] = { 0, };
int host_fsymtbl[144] = { 0, };
int host_enexttbl[12] = { 0, };
int host_eprevtbl[12] = { 0, };
int host_enextesymtbl[12] = { 0, };
int host_eprevesymtbl[12] = { 0, };
int host_eorgoppotbl[12] = { 0, };
int host_edestoppotbl[12] = { 0, };
int host_facepivot1[12] = { 0, };
int host_facepivot2[144] = { 0, };
int host_tsbondtbl[72] = { 0, };
int host_stbondtbl[72] = { 0, };
int host_tspivottbl[72] = { 0, };
int host_stpivottbl[72] = { 0, };

// Table 'esymtbl' takes an directed edge (version) as input, returns the
//   inversed edge (version) of it.

__constant__ int raw_esymtbl[12];

int host_esymtbl[12] = { 9, 6, 11, 4, 3, 7, 1, 5, 10, 0, 8, 2 };

// The following four tables give the 12 permutations of the set {0,1,2,3}.

__constant__ int raw_orgpivot[12];
__constant__ int raw_destpivot[12];
__constant__ int raw_apexpivot[12];
__constant__ int raw_oppopivot[12];

int host_orgpivot[12] = { 3, 3, 1, 1, 2, 0, 0, 2, 1, 2, 3, 0 };
int host_destpivot[12] = { 2, 0, 0, 2, 1, 2, 3, 0, 3, 3, 1, 1 };
int host_apexpivot[12] = { 1, 2, 3, 0, 3, 3, 1, 1, 2, 0, 0, 2 };
int host_oppopivot[12] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };

// The twelve versions correspond to six undirected edges. The following two
//   tables map a version to an undirected edge and vice versa.

__constant__ int raw_ver2edge[12];
__constant__ int raw_edge2ver[6];

int host_ver2edge[12] = { 0, 1, 2, 3, 3, 5, 1, 5, 4, 0, 4, 2 };
int host_edge2ver[6] = { 0, 1, 2, 3, 8, 5 };

// Edge versions whose apex or opposite may be dummypoint.

__constant__ int raw_epivot[12];

int host_epivot[12] = { 4, 5, 2, 11, 4, 5, 2, 11, 4, 5, 2, 11 };

// Table 'snextpivot' takes an edge version as input, returns the next edge
//   version in the same edge ring.

__constant__ int raw_snextpivot[6];

int host_snextpivot[6] = { 2, 5, 4, 1, 0, 3 };

// The following three tables give the 6 permutations of the set {0,1,2}.
//   An offset 3 is added to each element for a direct access of the points
//   in the triangle data structure.

__constant__ int raw_sorgpivot[6];
__constant__ int raw_sdestpivot[6];
__constant__ int raw_sapexpivot[6];


int host_sorgpivot[6] = { 0, 1, 1, 2, 2, 0 };
int host_sdestpivot[6] = { 1, 0, 2, 1, 0, 2 };
int host_sapexpivot[6] = { 2, 2, 0, 0, 1, 1 };

/* Initialize Geometric Predicates arrays*/

REAL host_constData[17];
int host_constOptions[2];

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric helpers														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ bool cudamesh_lu_decmp(REAL lu[4][4], int n, int* ps, REAL* d, int N)
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

__device__ void cudamesh_lu_solve(REAL lu[4][4], int n, int* ps, REAL* b, int N)
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

__device__ bool cudamesh_circumsphere(REAL* pa, REAL* pb, REAL* pc, REAL* pd,
	REAL* cent, REAL* radius)
{
	REAL A[4][4], rhs[4], D;
	int indx[4];

	// Compute the coefficient matrix A (3x3).
	A[0][0] = pb[0] - pa[0];
	A[0][1] = pb[1] - pa[1];
	A[0][2] = pb[2] - pa[2];
	A[1][0] = pc[0] - pa[0];
	A[1][1] = pc[1] - pa[1];
	A[1][2] = pc[2] - pa[2];
	if (pd != NULL) {
		A[2][0] = pd[0] - pa[0];
		A[2][1] = pd[1] - pa[1];
		A[2][2] = pd[2] - pa[2];
	}
	else {
		cudamesh_cross(A[0], A[1], A[2]);
	}

	// Compute the right hand side vector b (3x1).
	rhs[0] = 0.5 * cudamesh_dot(A[0], A[0]);
	rhs[1] = 0.5 * cudamesh_dot(A[1], A[1]);
	if (pd != NULL) {
		rhs[2] = 0.5 * cudamesh_dot(A[2], A[2]);
	}
	else {
		rhs[2] = 0.0;
	}

	// Solve the 3 by 3 equations use LU decomposition with partial pivoting
	//   and backward and forward substitute..
	if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
		if (radius != (REAL *)NULL) *radius = 0.0;
		return false;
	}
	cudamesh_lu_solve(A, 3, indx, rhs, 0);
	if (cent != (REAL *)NULL) {
		cent[0] = pa[0] + rhs[0];
		cent[1] = pa[1] + rhs[1];
		cent[2] = pa[2] + rhs[2];
	}
	if (radius != (REAL *)NULL) {
		*radius = sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
	}
	return true;
}

__device__ void cudamesh_facenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
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
		L1 = cudamesh_dot(v1, v1);
		L2 = cudamesh_dot(v2, v2);
		L3 = cudamesh_dot(v3, v3);
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
	cudamesh_cross(pv1, pv2, n);
	// Inverse the direction;
	n[0] = -n[0];
	n[1] = -n[1];
	n[2] = -n[2];
}

__device__ void cudamesh_calculateabovepoint4(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* abovept)
{
	REAL n1[3], n2[3], *norm;
	REAL len, len1, len2;

	// Select a base.
	cudamesh_facenormal(pa, pb, pc, n1, 1, NULL);
	len1 = sqrt(cudamesh_dot(n1, n1));
	cudamesh_facenormal(pa, pb, pd, n2, 1, NULL);
	len2 = sqrt(cudamesh_dot(n2, n2));
	if (len1 > len2) {
		norm = n1;
		len = len1;
	}
	else {
		norm = n2;
		len = len2;
	}
	assert(len > 0);
	norm[0] /= len;
	norm[1] /= len;
	norm[2] /= len;
	len = cudamesh_distance(pa, pb);
	abovept[0] = pa[0] + len * norm[0];
	abovept[1] = pa[1] + len * norm[1];
	abovept[2] = pa[2] + len * norm[2];
}

__device__ int cudamesh_segsegadjacent(
	int seg1,
	int seg2,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist
)
{
	int segidx1 = d_seg2parentidxlist[seg1];
	int segidx2 = d_seg2parentidxlist[seg2];

	if (segidx1 == segidx2)
		return 0;

	int pa1 = d_segparentendpointidxlist[segidx1 * 2];
	int pb1 = d_segparentendpointidxlist[segidx1 * 2 + 1];
	int pa2 = d_segparentendpointidxlist[segidx2 * 2];
	int pb2 = d_segparentendpointidxlist[segidx2 * 2 + 1];

	if ((pa1 == pa2) || (pa1 == pb2) || (pb1 == pa2) || (pb1 == pb2))
		return 1;
	return 0;
}

__device__ int cudamesh_segfacetadjacent(
	int subseg,
	int subsh,
	int* d_seg2parentidxlist,
	int* d_segparentendpointidxlist,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist
)
{
	int segidx = d_seg2parentidxlist[subseg];
	int pa = d_segparentendpointidxlist[segidx * 2];
	int pb = d_segparentendpointidxlist[segidx * 2 + 1];

	int fidx = d_tri2parentidxlist[subsh];
	int count = 0, i;

	int p;
	for (i = d_triid2parentoffsetlist[fidx]; i < d_triid2parentoffsetlist[fidx + 1]; i++)
	{
		p = d_triparentendpointidxlist[i];
		if (p == pa || p == pb)
			count++;
	}

	return count == 1;
}

__device__ int cudamesh_facetfacetadjacent(
	int subsh1,
	int subsh2,
	int* d_tri2parentidxlist,
	int* d_triid2parentoffsetlist,
	int* d_triparentendpointidxlist
)
{
	int count = 0;

	int fidx1 = d_tri2parentidxlist[subsh1];
	int fidx2 = d_tri2parentidxlist[subsh2];

	if (fidx1 == fidx2) return 0;

	int p1, p2;
	for (int i = d_triid2parentoffsetlist[fidx1]; i < d_triid2parentoffsetlist[fidx1 + 1]; i++)
	{
		p1 = d_triparentendpointidxlist[i];
		for (int j = d_triid2parentoffsetlist[fidx2]; j < d_triid2parentoffsetlist[fidx2 + 1]; j++)
		{
			p2 = d_triparentendpointidxlist[j];
			if (p1 == p2)
			{
				count++;
				break;
			}
		}
	}

	return count > 0;
}

__device__ REAL cudamesh_triangle_squared_area(
	REAL* pa, REAL* pb, REAL* pc
)
{
	// Compute the area of this 3D triangle
	REAL AB[3], AC[3];
	int i;
	for (i = 0; i < 3; i++)
	{
		AB[i] = pb[i] - pa[i];
		AC[i] = pc[i] - pa[i];
	}

	REAL sarea =
		((AB[1] * AC[2] - AB[2] * AC[1])*(AB[1] * AC[2] - AB[2] * AC[1]) +
		(AB[2] * AC[0] - AB[0] * AC[2])*(AB[2] * AC[0] - AB[0] * AC[2]) +
			(AB[0] * AC[1] - AB[1] * AC[0])*(AB[0] * AC[1] - AB[1] * AC[0])) / 4;

	return sarea;
}

__device__ REAL cudamesh_tetrahedronvolume(
	REAL* pa, REAL* pb, REAL* pc, REAL* pd
)
{
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];

	int i;
	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	for (i = 0; i < 3; i++) vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	elen[0] = cudamesh_dot(vda, vda);
	elen[1] = cudamesh_dot(vdb, vdb);
	elen[2] = cudamesh_dot(vdc, vdc);
	elen[3] = cudamesh_dot(vab, vab);
	elen[4] = cudamesh_dot(vbc, vbc);
	elen[5] = cudamesh_dot(vca, vca);

	// Use heron-type formula to compute the volume of a tetrahedron
	// https://en.wikipedia.org/wiki/Heron%27s_formula
	REAL U, V, W, u, v, w; // first three form a triangle; u opposite to U and so on
	REAL X, x, Y, y, Z, z;
	REAL a, b, c, d;
	U = sqrt(elen[3]); //ab
	V = sqrt(elen[4]); //bc
	W = sqrt(elen[5]); //ca
	u = sqrt(elen[2]); //dc
	v = sqrt(elen[0]); //da
	w = sqrt(elen[1]); //db

	X = (w - U + v)*(U + v + w);
	x = (U - v + w)*(v - w + U);
	Y = (u - V + w)*(V + w + u);
	y = (V - w + u)*(w - u + V);
	Z = (v - W + u)*(W + u + v);
	z = (W - u + v)*(u - v + W);

	a = sqrt(x*Y*Z);
	b = sqrt(y*Z*X);
	c = sqrt(z*X*Y);
	d = sqrt(x*y*z);

	REAL vol, val1, val2;
	val1 = (-a + b + c + d)*(a - b + c + d)*(a + b - c + d)*(a + b + c - d);
	val2 = 192 * u*v*w;
	if (val1 < 0.0 || val2 == 0.0)
		vol = 0.0;
	else
		vol = sqrt(val1) / val2;
	return vol;
}

__device__ REAL cudamesh_tetrahedronvolume(
	int tetid,
	REAL* d_pointlist,
	int* d_tetlist
)
{
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];

	int ipa, ipb, ipc, ipd;
	REAL *pa, *pb, *pc, *pd;
	int i;

	ipa = d_tetlist[4 * tetid + 0];
	ipb = d_tetlist[4 * tetid + 1];
	ipc = d_tetlist[4 * tetid + 2];
	ipd = d_tetlist[4 * tetid + 3];

	pa = cudamesh_id2pointlist(ipa, d_pointlist);
	pb = cudamesh_id2pointlist(ipb, d_pointlist);
	pc = cudamesh_id2pointlist(ipc, d_pointlist);
	pd = cudamesh_id2pointlist(ipd, d_pointlist);

	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	for (i = 0; i < 3; i++) vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	elen[0] = cudamesh_dot(vda, vda);
	elen[1] = cudamesh_dot(vdb, vdb);
	elen[2] = cudamesh_dot(vdc, vdc);
	elen[3] = cudamesh_dot(vab, vab);
	elen[4] = cudamesh_dot(vbc, vbc);
	elen[5] = cudamesh_dot(vca, vca);

	// Use heron-type formula to compute the volume of a tetrahedron
	// https://en.wikipedia.org/wiki/Heron%27s_formula
	REAL U, V, W, u, v, w; // first three form a triangle; u opposite to U and so on
	REAL X, x, Y, y, Z, z;
	REAL a, b, c, d;
	U = sqrt(elen[3]); //ab
	V = sqrt(elen[4]); //bc
	W = sqrt(elen[5]); //ca
	u = sqrt(elen[2]); //dc
	v = sqrt(elen[0]); //da
	w = sqrt(elen[1]); //db

	X = (w - U + v)*(U + v + w);
	x = (U - v + w)*(v - w + U);
	Y = (u - V + w)*(V + w + u);
	y = (V - w + u)*(w - u + V);
	Z = (v - W + u)*(W + u + v);
	z = (W - u + v)*(u - v + W);

	a = sqrt(x*Y*Z);
	b = sqrt(y*Z*X);
	c = sqrt(z*X*Y);
	d = sqrt(x*y*z);

	REAL vol = sqrt((-a + b + c + d)*(a - b + c + d)*(a + b - c + d)*(a + b + c - d)) / (192 * u*v*w);
	return vol;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Geometric predicates with symbolic perturbation							 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

__device__ REAL cudamesh_insphere_s(REAL* pa, REAL* pb, REAL* pc, REAL* pd, REAL* pe, 
	int ia, int ib, int ic, int id, int ie)
{
	REAL sign;
	// Using fast version means using inexact method.
	// This may cause robustness issues.
	// Need to handle later on.
	sign = cuda_inspherefast(pa, pb, pc, pd, pe);
	//if (fabs(sign) < EPSILON)
	//	sign = cuda_insphereexact(pa, pb, pc, pd, pe);

	if (sign != 0.0) {
		return sign;
	}

	// Symbolic perturbation.
	REAL* pt[5], *swappt;
	int idx[5], swapidx;
	REAL oriA, oriB;
	int swaps, count;
	int n, i;

	pt[0] = pa;
	pt[1] = pb;
	pt[2] = pc;
	pt[3] = pd;
	pt[4] = pe;

	idx[0] = ia;
	idx[1] = ib;
	idx[2] = ic;
	idx[3] = id;
	idx[4] = ie;

	// Sort the five points such that their indices are in the increasing
	//   order. An optimized bubble sort algorithm is used, i.e., it has
	//   the worst case O(n^2) runtime, but it is usually much faster.
	swaps = 0; // Record the total number of swaps.
	n = 5;
	do {
		count = 0;
		n = n - 1;
		for (i = 0; i < n; i++) {
			if (idx[i] > idx[i + 1]) {
				swappt = pt[i]; pt[i] = pt[i + 1]; pt[i + 1] = swappt;
				swapidx = idx[i]; idx[i] = idx[i + 1]; idx[i + 1] = swapidx;
				count++;
			}
		}
		swaps += count;
		break;
	} while (count > 0); // Continue if some points are swapped.

	oriA = cuda_orient3d(pt[1], pt[2], pt[3], pt[4]);
	if (oriA != 0.0) {
		// Flip the sign if there are odd number of swaps.
		if ((swaps % 2) != 0) oriA = -oriA;
		return oriA;
	}

	oriB = -cuda_orient3d(pt[0], pt[2], pt[3], pt[4]);
	assert(oriB != 0.0); // SELF_CHECK
						 // Flip the sign if there are odd number of swaps.
	if ((swaps % 2) != 0) oriB = -oriB;
	return oriB;
}

__device__ REAL cudamesh_incircle3d(REAL* pa, REAL* pb, REAL* pc, REAL* pd)
{
	REAL area2[2], n1[3], n2[3], c[3];
	REAL sign, r, d;

	// Calculate the areas of the two triangles [a, b, c] and [b, a, d].
	cudamesh_facenormal(pa, pb, pc, n1, 1, NULL);
	area2[0] = cudamesh_dot(n1, n1);
	cudamesh_facenormal(pb, pa, pd, n2, 1, NULL);
	area2[1] = cudamesh_dot(n2, n2);

	if (area2[0] > area2[1]) {
		// Choose [a, b, c] as the base triangle.
		cudamesh_circumsphere(pa, pb, pc, NULL, c, &r);
		d = cudamesh_distance(c, pd);
	}
	else {
		// Choose [b, a, d] as the base triangle.
		if (area2[1] > 0) {
			cudamesh_circumsphere(pb, pa, pd, NULL, c, &r);
			d = cudamesh_distance(c, pc);
		}
		else {
			// The four points are collinear. This case only happens on the boundary.
			return 0; // Return "not inside".
		}
	}

	sign = d - r;
	if (fabs(sign) / r < EPSILON) {
		sign = 0;
	}

	return sign;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Mesh manipulation primitives                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/* Initialize tables */
void cudamesh_inittables()
{
	// init arrays
	int i, j;

	cudaMemcpyToSymbol(raw_esymtbl, host_esymtbl, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_orgpivot, host_orgpivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_destpivot, host_destpivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_apexpivot, host_apexpivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_oppopivot, host_oppopivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_ver2edge, host_ver2edge, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_edge2ver, host_edge2ver, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_epivot, host_epivot, 12 * sizeof(int));

	cudaMemcpyToSymbol(raw_snextpivot, host_snextpivot, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_sorgpivot, host_sorgpivot, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_sdestpivot, host_sdestpivot, 6 * sizeof(int));

	cudaMemcpyToSymbol(raw_sapexpivot, host_sapexpivot, 6 * sizeof(int));

	// i = t1.ver; j = t2.ver;
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			host_bondtbl[12* i + j] = (j & 3) + (((i & 12) + (j & 12)) % 12);
		}
	}
	cudaMemcpyToSymbol(raw_bondtbl, host_bondtbl, 144 * sizeof(int));

	// i = t1.ver; j = t2.ver
	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			host_fsymtbl[12 * i + j] = (j + 12 - (i & 12)) % 12;
		}
	}
	cudaMemcpyToSymbol(raw_fsymtbl, host_fsymtbl, 144 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_facepivot1[i] = (host_esymtbl[i] & 3);
	}
	cudaMemcpyToSymbol(raw_facepivot1, host_facepivot1, 12 * sizeof(int));

	for (i = 0; i < 12; i++) {
		for (j = 0; j < 12; j++) {
			host_facepivot2[12 * i + j] = host_fsymtbl[12 * host_esymtbl[i] + j];
		}
	}
	cudaMemcpyToSymbol(raw_facepivot2, host_facepivot2, 144 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_enexttbl[i] = (i + 4) % 12;
		host_eprevtbl[i] = (i + 8) % 12;
	}
	cudaMemcpyToSymbol(raw_enexttbl, host_enexttbl, 12 * sizeof(int));
	cudaMemcpyToSymbol(raw_eprevtbl, host_eprevtbl, 12 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_enextesymtbl[i] = host_esymtbl[host_enexttbl[i]];
		host_eprevesymtbl[i] = host_esymtbl[host_eprevtbl[i]];
	}
	cudaMemcpyToSymbol(raw_enextesymtbl, host_enextesymtbl, 12 * sizeof(int));
	cudaMemcpyToSymbol(raw_eprevesymtbl, host_eprevesymtbl, 12 * sizeof(int));

	for (i = 0; i < 12; i++) {
		host_eorgoppotbl[i] = host_eprevtbl[host_esymtbl[host_enexttbl[i]]];
		host_edestoppotbl[i] = host_enexttbl[host_esymtbl[host_eprevtbl[i]]];
	}
	cudaMemcpyToSymbol(raw_eorgoppotbl, host_eorgoppotbl, 12 * sizeof(int));
	cudaMemcpyToSymbol(raw_edestoppotbl, host_edestoppotbl, 12 * sizeof(int));

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
			host_tsbondtbl[6 * i + j] = (j & 1) + (((j & 6) + soffset) % 6);
			host_stbondtbl[6 * i + j] = (i & 3) + (((i & 12) + toffset) % 12);
		}
	}
	cudaMemcpyToSymbol(raw_tsbondtbl, host_tsbondtbl, 72 * sizeof(int));
	cudaMemcpyToSymbol(raw_stbondtbl, host_stbondtbl, 72 * sizeof(int));

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
			host_tspivottbl[6 * i + j] = (j & 1) + (((j & 6) + soffset) % 6);
			host_stpivottbl[6 * i + j] = (i & 3) + (((i & 12) + toffset) % 12);
		}
	}
	cudaMemcpyToSymbol(raw_tspivottbl, host_tspivottbl, 72 * sizeof(int));
	cudaMemcpyToSymbol(raw_stpivottbl, host_stpivottbl, 72 * sizeof(int));
}


/* Init bounding box*/
void cudamesh_initbbox(
	int numofpoints, double* pointlist,
	int& xmax, int& xmin, int& ymax, int& ymin, int& zmax, int& zmin)
{
	int i;
	double x, y, z;
	for (i = 0; i < numofpoints; i++)
	{
		x = pointlist[3 * i];
		y = pointlist[3 * i + 1];
		z = pointlist[3 * i + 2];
		if (i == 0) 
		{
			xmin = xmax = x;
			ymin = ymax = y;
			zmin = zmax = z;
		}
		else
		{
			xmin = (x < xmin) ? x : xmin;
			xmax = (x > xmax) ? x : xmax;
			ymin = (y < ymin) ? y : ymin;
			ymax = (y > ymax) ? y : ymax;
			zmin = (z < zmin) ? z : zmin;
			zmax = (z > zmax) ? z : zmax;
		}
	}
}

/* Initialize Geometric primitives */
void cudamesh_exactinit(int verbose, int noexact, int nofilter,
	REAL maxx, REAL maxy, REAL maxz)
{
	REAL half;
	REAL check, lastcheck;
	int every_other;

	every_other = 1;
	half = 0.5;
	host_constData[1] /*epsilon*/ = 1.0;
	host_constData[0] /*splitter*/ = 1.0;
	check = 1.0;
	/* Repeatedly divide `epsilon' by two until it is too small to add to    */
	/*   one without causing roundoff.  (Also check if the sum is equal to   */
	/*   the previous sum, for machines that round up instead of using exact */
	/*   rounding.  Not that this library will work on such machines anyway. */
	do {
		lastcheck = check;
		host_constData[1] /*epsilon*/ *= half;
		if (every_other) {
			host_constData[0] /*splitter*/ *= 2.0;
		}
		every_other = !every_other;
		check = 1.0 +  host_constData[1] /*epsilon*/;
	} while ((check != 1.0) && (check != lastcheck));
	host_constData[0] /*splitter*/ += 1.0;

	/* Error bounds for orientation and incircle tests. */
	host_constData[2] /*resulterrbound*/ = (3.0 + 8.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[3] /*ccwerrboundA*/ = (3.0 + 16.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[4] /*ccwerrboundB*/ = (2.0 + 12.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[5] /*ccwerrboundC*/ = (9.0 + 64.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[6] /*o3derrboundA*/ = (7.0 + 56.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[7] /*o3derrboundB*/ = (3.0 + 28.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[8] /*o3derrboundC*/ = (26.0 + 288.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[9] /*iccerrboundA*/ = (10.0 + 96.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[10] /*iccerrboundB*/ = (4.0 + 48.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[11] /*iccerrboundC*/ = (44.0 + 576.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[12] /*isperrboundA*/ = (16.0 + 224.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[13] /*isperrboundB*/ = (5.0 + 72.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[14] /*isperrboundC*/ = (71.0 + 1408.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;

	// Set TetGen options.  Added by H. Si, 2012-08-23.
	host_constOptions[0] /*_use_inexact_arith*/ = noexact;
	host_constOptions[1] /*_use_static_filter*/ = !nofilter;

	// Calculate the two static filters for orient3d() and insphere() tests.
	// Added by H. Si, 2012-08-23.

	// Sort maxx < maxy < maxz. Re-use 'half' for swapping.
	assert(maxx > 0);
	assert(maxy > 0);
	assert(maxz > 0);

	if (maxx > maxz) {
		half = maxx; maxx = maxz; maxz = half;
	}
	if (maxy > maxz) {
		half = maxy; maxy = maxz; maxz = half;
	}
	else if (maxy < maxx) {
		half = maxy; maxy = maxx; maxx = half;
	}

	host_constData[15] /*o3dstaticfilter*/ = 5.1107127829973299e-15 * maxx * maxy * maxz;
	host_constData[16] /*ispstaticfilter*/ = 1.2466136531027298e-13 * maxx * maxy * maxz * (maxz * maxz);

	// Copy to const memory
	cudaMemcpyToSymbol(raw_constData, host_constData, 17 * sizeof(REAL));
	cudaMemcpyToSymbol(raw_constOptions, host_constOptions, 2 * sizeof(int));

	//for (int i = 0; i<17; i++)
	//	printf("host_constData[%d] = %g\n", i, host_constData[i]);
	//for (int i = 0; i < 2; i++)
	//	printf("host_constOptions[%d] = %d\n", i, host_constOptions[i]);
}

/* Init Kernel constants */
void cudamesh_initkernelconstants(REAL maxx, REAL maxy, REAL maxz)
{
	REAL longest = sqrt(maxx*maxx + maxy*maxy + maxz*maxz);
	REAL minedgelength = longest*EPSILON;
	host_kernelconstants[0] = minedgelength;

	cudaMemcpyToSymbol(raw_kernelconstants, host_kernelconstants, sizeof(REAL));
}

/* Primitives for points */

// Convert point index to pointer to pointlist
__device__ double* cudamesh_id2pointlist(int index, double* pointlist)
{
	return (pointlist + 3 * index);
}


/* Primitives for tetrahedron */

// The following primtives get or set the origin, destination, face apex,
//   or face opposite of an ordered tetrahedron.

__device__ int cudamesh_org(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_orgpivot[t.ver]];
}

__device__ int cudamesh_dest(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_destpivot[t.ver]];
}

__device__ int cudamesh_apex(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_apexpivot[t.ver]];
}

__device__ int cudamesh_oppo(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + raw_oppopivot[t.ver]];
}

__device__ void cudamesh_setorg(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_orgpivot[t.ver]] = p;
}

__device__ void cudamesh_setdest(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_destpivot[t.ver]] = p;
}

__device__ void cudamesh_setapex(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_apexpivot[t.ver]] = p;
}

__device__ void cudamesh_setoppo(tethandle t, int p, int* tetlist)
{
	tetlist[4 * t.id + raw_oppopivot[t.ver]] = p;
}

// bond()  connects two tetrahedra together. (t1,v1) and (t2,v2) must 
//   refer to the same face and the same edge.

__device__ void cudamesh_bond(tethandle t1, tethandle t2, tethandle* neighborlist)
{
	neighborlist[4 * t1.id + (t1.ver & 3)] = tethandle(t2.id, raw_bondtbl[12 * t1.ver + t2.ver]);
	neighborlist[4 * t2.id + (t2.ver & 3)] = tethandle(t1.id, raw_bondtbl[12 * t2.ver + t1.ver]);
}

// dissolve()  a bond (from one side).

__device__ void cudamesh_dissolve(tethandle t, tethandle* neighborlist)
{
	neighborlist[4 * t.id + (t.ver & 3)] = tethandle(-1, 11); // empty handle
}

// esym()  finds the reversed edge.  It is in the other face of the
//   same tetrahedron.

__device__ void cudamesh_esym(tethandle& t1, tethandle& t2)
{
	(t2).id = (t1).id;
	(t2).ver = raw_esymtbl[(t1).ver];
}
__device__ void cudamesh_esymself(tethandle& t)
{
	(t).ver = raw_esymtbl[(t).ver];
}

// enext()  finds the next edge (counterclockwise) in the same face.

__device__ void cudamesh_enext(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = raw_enexttbl[t1.ver];
}
__device__ void cudamesh_enextself(tethandle& t)
{
	t.ver = raw_enexttbl[t.ver];
}

// eprev()   finds the next edge (clockwise) in the same face.

__device__ void cudamesh_eprev(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = raw_eprevtbl[t1.ver];
}
__device__ void cudamesh_eprevself(tethandle& t)
{
	t.ver = raw_eprevtbl[t.ver];
}

// enextesym()  finds the reversed edge of the next edge. It is in the other
//   face of the same tetrahedron. It is the combination esym() * enext(). 

__device__ void cudamesh_enextesym(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = raw_enextesymtbl[t1.ver];
}

__device__ void cudamesh_enextesymself(tethandle& t) {
	t.ver = raw_enextesymtbl[t.ver];
}

// eprevesym()  finds the reversed edge of the previous edge.

__device__ void cudamesh_eprevesym(tethandle& t1, tethandle& t2)
{
	t2.id = t1.id;
	t2.ver = raw_eprevesymtbl[t1.ver];
}

__device__ void cudamesh_eprevesymself(tethandle& t) {
	t.ver = raw_eprevesymtbl[t.ver];
}

// eorgoppo()    Finds the opposite face of the origin of the current edge.
//               Return the opposite edge of the current edge.

__device__ void cudamesh_eorgoppo(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = raw_eorgoppotbl[t1.ver];
}

__device__ void cudamesh_eorgoppoself(tethandle& t) {
	t.ver = raw_eorgoppotbl[t.ver];
}

// edestoppo()    Finds the opposite face of the destination of the current 
//                edge. Return the opposite edge of the current edge.

__device__ void cudamesh_edestoppo(tethandle& t1, tethandle& t2) {
	t2.id = t1.id;
	t2.ver = raw_edestoppotbl[t1.ver];
}

__device__ void cudamesh_edestoppoself(tethandle& t) {
	t.ver = raw_edestoppotbl[t.ver];
}

// fsym()  finds the adjacent tetrahedron at the same face and the same edge.

__device__ void cudamesh_fsym(tethandle& t1, tethandle& t2, tethandle* neighborlist)
{
	t2 = neighborlist[4 * t1.id + (t1.ver & 3)];
	t2.ver = raw_fsymtbl[12 * t1.ver + t2.ver];
}

__device__ void cudamesh_fsymself(tethandle& t, tethandle* neighborlist)
{
	char t1ver = t.ver;
	t = neighborlist[4 * t.id + (t.ver & 3)];
	t.ver = raw_fsymtbl[12 * t1ver + t.ver];
}

// fnext()  finds the next face while rotating about an edge according to
//   a right-hand rule. The face is in the adjacent tetrahedron.  It is
//   the combination: fsym() * esym().

__device__ void cudamesh_fnext(tethandle& t1, tethandle& t2, tethandle* neighborlist)
{
	t2 = neighborlist[4 * t1.id + raw_facepivot1[t1.ver]];
	t2.ver = raw_facepivot2[12 * t1.ver + t2.ver];
}

__device__ void cudamesh_fnextself(tethandle& t, tethandle* neighborlist)
{
	char t1ver = t.ver;
	t = neighborlist[4 * t.id + raw_facepivot1[t.ver]];
	t.ver = raw_facepivot2[12 * t1ver + t.ver];
}

// ishulltet()  tests if t is a hull tetrahedron.

__device__ bool cudamesh_ishulltet(tethandle t, int* tetlist)
{
	return tetlist[4 * t.id + 3] == -1;
}

// isdeadtet()  tests if t is a tetrahedron is dead.

__device__ bool cudamesh_isdeadtet(tethandle t)
{
	return (t.id == -1);
}

/* Primitives for subfaces and subsegments. */

// spivot() finds the adjacent subface (s2) for a given subface (s1).
//   s1 and s2 share at the same edge.

__device__ void cudamesh_spivot(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	s2 = tri2trilist[3 * s1.id + (s1.shver >> 1)];
}

__device__ void cudamesh_spivotself(trihandle& s, trihandle* tri2trilist)
{
	s = tri2trilist[3 * s.id + (s.shver >> 1)];
}

// sbond() bonds two subfaces (s1) and (s2) together. s1 and s2 must refer
//   to the same edge. No requirement is needed on their orientations.

__device__ void cudamesh_sbond(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	tri2trilist[3 * s1.id + (s1.shver >> 1)] = s2;
	tri2trilist[3 * s2.id + (s2.shver >> 1)] = s1;
}

// sbond1() bonds s1 <== s2, i.e., after bonding, s1 is pointing to s2,
//   but s2 is not pointing to s1.  s1 and s2 must refer to the same edge.
//   No requirement is needed on their orientations.

__device__ void cudamesh_sbond1(trihandle& s1, trihandle& s2, trihandle* tri2trilist)
{
	tri2trilist[3 * s1.id + (s1.shver >> 1)] = s2;
}

// Dissolve a subface bond (from one side).  Note that the other subface
//   will still think it's connected to this subface.
__device__ void cudamesh_sdissolve(trihandle& s, trihandle* tri2trilist)
{
	tri2trilist[3 * s.id + (s.shver >> 1)] = trihandle(-1, 0);
}

// These primitives determine or set the origin, destination, or apex
//   of a subface with respect to the edge version.

__device__ int cudamesh_sorg(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + raw_sorgpivot[s.shver]];
}

__device__ int cudamesh_sdest(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + raw_sdestpivot[s.shver]];
}

__device__ int cudamesh_sapex(trihandle& s, int* trilist)
{
	return trilist[3 * s.id + raw_sapexpivot[s.shver]];
}

__device__ void cudamesh_setsorg(trihandle& s, int p, int* trilist)
{
	trilist[3 * s.id + raw_sorgpivot[s.shver]] = p;
}

__device__ void cudamesh_setsdest(trihandle& s, int p, int* trilist)
{
	trilist[3 * s.id + raw_sdestpivot[s.shver]] = p;
}

__device__ void cudamesh_setsapex(trihandle& s, int p, int* trilist)
{
	trilist[3 * s.id + raw_sapexpivot[s.shver]] = p;
}

// sesym()  reserves the direction of the lead edge.

__device__ void cudamesh_sesym(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = (s1.shver ^ 1);  // Inverse the last bit.
}

__device__ void cudamesh_sesymself(trihandle& s)
{
	s.shver ^= 1;
}

// senext()  finds the next edge (counterclockwise) in the same orientation
//   of this face.

__device__ void cudamesh_senext(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = raw_snextpivot[s1.shver];
}

__device__ void cudamesh_senextself(trihandle& s)
{
	s.shver = raw_snextpivot[s.shver];
}

__device__ void cudamesh_senext2(trihandle& s1, trihandle& s2)
{
	s2.id = s1.id;
	s2.shver = raw_snextpivot[raw_snextpivot[s1.shver]];
}

__device__ void cudamesh_senext2self(trihandle& s)
{
	s.shver = raw_snextpivot[raw_snextpivot[s.shver]];
}


/* Primitives for interacting tetrahedra and subfaces. */

// tsbond() bond a tetrahedron (t) and a subface (s) together.
// Note that t and s must be the same face and the same edge. Moreover,
//   t and s have the same orientation. 
// Since the edge number in t and in s can be any number in {0,1,2}. We bond
//   the edge in s which corresponds to t's 0th edge, and vice versa.

__device__ void cudamesh_tsbond(tethandle& t, trihandle& s, trihandle* tet2trilist, tethandle* tri2tetlist)
{
	// Bond t <== s.
	tet2trilist[4 * t.id + (t.ver & 3)] = trihandle(s.id, raw_tsbondtbl[6 * t.ver + s.shver]);
	// Bond s <== t.
	tri2tetlist[2 * s.id + (s.shver & 1)] = tethandle(t.id, raw_stbondtbl[6 * t.ver + s.shver]);
}

// tspivot() finds a subface (s) abutting on the given tetrahdera (t).
//   Return s.id = -1 if there is no subface at t. Otherwise, return
//   the subface s, and s and t must be at the same edge wth the same
//   orientation.
__device__ void cudamesh_tspivot(tethandle& t, trihandle& s, trihandle* tet2trilist)
{
	// Get the attached subface s.
	s = tet2trilist[4 * t.id + (t.ver & 3)];
	if (s.id == -1)
		return;
	(s).shver = raw_tspivottbl[6 * t.ver + s.shver];
}

// stpivot() finds a tetrahedron (t) abutting a given subface (s).
//   Return the t (if it exists) with the same edge and the same
//   orientation of s.
__device__ void cudamesh_stpivot(trihandle& s, tethandle& t, tethandle* tri2tetlist)
{
	t = tri2tetlist[2 * s.id + (s.shver & 1)];
	if (t.id == -1) {
		return;
	}
	(t).ver = raw_stpivottbl[6 * t.ver + s.shver];
}

/* Primitives for interacting between tetrahedra and segments */

__device__ void cudamesh_tsspivot1(tethandle& t, trihandle& seg, trihandle* tet2seglist)
{
	seg = tet2seglist[6 * t.id + raw_ver2edge[t.ver]];
}

__device__ void cudamesh_tssbond1(tethandle& t, trihandle& seg, trihandle* tet2seglist)
{
	tet2seglist[6 * t.id + raw_ver2edge[t.ver]] = seg;
}

__device__ void cudamesh_sstbond1(trihandle& s, tethandle& t, tethandle* seg2tetlist)
{
	seg2tetlist[s.id + 0] = t;
}

__device__ void cudamesh_sstpivot1(trihandle& s, tethandle& t, tethandle* seg2tetlist)
{
	t = seg2tetlist[s.id];
}

/* Primitives for interacting between subfaces and segments */

__device__ void cudamesh_ssbond(trihandle& s, trihandle& edge, trihandle* tri2seglist, trihandle* seg2trilist)
{
	tri2seglist[3 * s.id + (s.shver >> 1)] = edge;
	seg2trilist[3 * edge.id + 0] = s;
}

__device__ void cudamesh_ssbond1(trihandle& s, trihandle& edge, trihandle* tri2seglist)
{
	tri2seglist[3 * s.id + (s.shver >> 1)] = edge;
}

__device__ void cudamesh_sspivot(trihandle& s, trihandle& edge, trihandle* tri2seglist)
{
	edge = tri2seglist[3 * s.id + (s.shver >> 1)];
}

__device__ bool cudamesh_isshsubseg(trihandle&s, trihandle* tri2seglist)
{
	return (tri2seglist[3 * s.id + (s.shver >> 1)].id != -1);
}



/* Advanced primitives. */

__device__ void cudamesh_point2tetorg(int pa, tethandle& searchtet, tethandle* point2tetlist, int* tetlist)
{
	searchtet = point2tetlist[pa];
	if (tetlist[4 * searchtet.id + 0] == pa) {
		searchtet.ver = 11;
	}
	else if (tetlist[4 * searchtet.id + 1] == pa) {
		searchtet.ver = 3;
	}
	else if (tetlist[4 * searchtet.id + 2] == pa) {
		searchtet.ver = 7;
	}
	else {
		assert(tetlist[4 * searchtet.id + 3] == pa); // SELF_CHECK
		searchtet.ver = 0;
	}
}

/* Geometric calculations (non-robust) */

// dot() returns the dot product: v1 dot v2.
__device__ REAL cudamesh_dot(REAL* v1, REAL* v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// distance() computes the Euclidean distance between two points.
__device__ REAL cudamesh_distance(REAL* p1, REAL* p2)
{
	//printf("%lf %lf %lf - %lf %lf %lf\n", 
	//	p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);

	return sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) +
		(p2[1] - p1[1]) * (p2[1] - p1[1]) +
		(p2[2] - p1[2]) * (p2[2] - p1[2]));
}

// cross() computes the cross product: n = v1 cross v2.
__device__ void cudamesh_cross(REAL* v1, REAL* v2, REAL* n)
{
	n[0] = v1[1] * v2[2] - v2[1] * v1[2];
	n[1] = -(v1[0] * v2[2] - v2[0] * v1[2]);
	n[2] = v1[0] * v2[1] - v2[0] * v1[1];
}

/* Helpers */

__device__ unsigned long cudamesh_randomnation(unsigned long * randomseed, unsigned int choices)
{
	unsigned long newrandom;

	if (choices >= 714025l) {
		newrandom = (*randomseed * 1366l + 150889l) % 714025l;
		*randomseed = (newrandom * 1366l + 150889l) % 714025l;
		newrandom = newrandom * (choices / 714025l) + *randomseed;
		if (newrandom >= choices) {
			return newrandom - choices;
		}
		else {
			return newrandom;
		}
	}
	else {
		*randomseed = (*randomseed * 1366l + 150889l) % 714025l;
		return *randomseed % choices;
	}
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

__device__ enum interresult cudamesh_finddirection(tethandle* searchtet, int endpt, double* pointlist, int* tetlist, tethandle* neighborlist, unsigned long* randomseed)
{
	tethandle neightet;
	int pa, pb, pc, pd;
	enum { HMOVE, RMOVE, LMOVE } nextmove;
	REAL hori, rori, lori;
	int t1ver;
	int s;

	// The origin is fixed.
	pa = cudamesh_org(*searchtet, tetlist);
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

	pb = cudamesh_dest(*searchtet, tetlist);
	// Check whether the destination or apex is 'endpt'.
	if (pb == endpt) {
		// pa->pb is the search edge.
		return ACROSSVERT;
	}

	pc = cudamesh_apex(*searchtet, tetlist);
	if (pc == endpt) {
		// pa->pc is the search edge.
		cudamesh_eprevesymself(*searchtet);
		return ACROSSVERT;
	}

	double *p[5];

	// Walk through tets around pa until the right one is found.
	while (1) {

		pd = cudamesh_oppo(*searchtet, tetlist);
		// Check whether the opposite vertex is 'endpt'.
		if (pd == endpt) {
			// pa->pd is the search edge.
			cudamesh_esymself(*searchtet);
			cudamesh_enextself(*searchtet);
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
		p[0] = cudamesh_id2pointlist(pa, pointlist);
		p[1] = cudamesh_id2pointlist(pb, pointlist);
		p[2] = cudamesh_id2pointlist(pc, pointlist);
		p[3] = cudamesh_id2pointlist(pd, pointlist);
		p[4] = cudamesh_id2pointlist(endpt, pointlist);

		hori = cuda_orient3d(p[0], p[1], p[2], p[4]);
		rori = cuda_orient3d(p[1], p[0], p[3], p[4]);
		lori = cuda_orient3d(p[0], p[2], p[3], p[4]);

		// Now decide the tet to move.  It is possible there are more than one
		//   tets are viable moves. Is so, randomly choose one. 
		if (hori > 0) {
			if (rori > 0) {
				if (lori > 0) {
					// Any of the three neighbors is a viable move.
					s = cudamesh_randomnation(randomseed, 3);
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
					if (cudamesh_randomnation(randomseed, 2)) {
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
					if (cudamesh_randomnation(randomseed, 2)) {
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
					if (cudamesh_randomnation(randomseed, 2)) {
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
							cudamesh_eprevesymself(*searchtet); // // [a,c,d]
							return ACROSSVERT;
						}
						// pa->'endpt' crosses the edge pb->pc.
						return ACROSSEDGE;
					}
					if (rori == 0) {
						if (lori == 0) {
							// pa->'endpt' is COLLINEAR with pa->pd.
							cudamesh_esymself(*searchtet); // face bad.
							cudamesh_enextself(*searchtet); // face [a,d,b]
							return ACROSSVERT;
						}
						// pa->'endpt' crosses the edge pb->pd.
						cudamesh_esymself(*searchtet); // face bad.
						cudamesh_enextself(*searchtet); // face adb
						return ACROSSEDGE;
					}
					if (lori == 0) {
						// pa->'endpt' crosses the edge pc->pd.
						cudamesh_eprevesymself(*searchtet); // [a,c,d]
						return ACROSSEDGE;
					}
					// pa->'endpt' crosses the face bcd.
					return ACROSSFACE;
				}
			}
		}

		// Move to the next tet, fix pa as its origin.
		if (nextmove == RMOVE) {
			cudamesh_fnextself(*searchtet, neighborlist);
		}
		else if (nextmove == LMOVE) {
			cudamesh_eprevself(*searchtet);
			cudamesh_fnextself(*searchtet, neighborlist);
			cudamesh_enextself(*searchtet);
		}
		else { // HMOVE
			cudamesh_fsymself(*searchtet, neighborlist);
			cudamesh_enextself(*searchtet);
		}
		assert(cudamesh_org(*searchtet, tetlist) == pa);
		pb = cudamesh_dest(*searchtet, tetlist);
		pc = cudamesh_apex(*searchtet, tetlist);

	} // while (1)
}

/////////////////////////////////////////////////////////////////////////////////
////                                                                           //
//// getedge()    Get a tetrahedron having the two endpoints.                  //
////                                                                           //
//// The method here is to search the second vertex in the link faces of the   //
//// first vertex. The global array 'cavetetlist' is re-used for searching.    //
////                                                                           //
//// This function is used for the case when the mesh is non-convex. Otherwise,//
//// the function finddirection() should be faster than this.                  //
////                                                                           //
/////////////////////////////////////////////////////////////////////////////////
//
////int getedge(int e1, int e2, tethandle *tedge, tethandle* point2tet, double* pointlist, int* tetlist, tethandle* neighborlist, int* markerlist)
////{
////	tethandle searchtet, neightet, parytet;
////	int pt;
////	int done;
////	int i, j;
////
////	// Quickly check if 'tedge' is just this edge.
////	if (!isdeadtet(*tedge)) {
////		if (org(*tedge, tetlist) == e1) {
////			if (dest(*tedge, tetlist) == e2) {
////				return 1;
////			}
////		}
////		else if (org(*tedge, tetlist) == e2) {
////			if (dest(*tedge, tetlist) == e1) {
////				esymself(*tedge);
////				return 1;
////			}
////		}
////	}
////
////	// Search for the edge [e1, e2].
////	point2tetorg(e1, *tedge, point2tet, tetlist);
////	finddirection(tedge, e2, pointlist, tetlist, neighborlist);
////	if (dest(*tedge, tetlist) == e2)
////	{
////		return 1;
////	}
////	else
////	{
////		// Search for the edge [e2, e1].
////		point2tetorg(e2, *tedge, point2tet, tetlist);
////		finddirection(tedge, e1, pointlist, tetlist, neighborlist);
////		if (dest(*tedge, tetlist) == e1) {
////			esymself(*tedge);
////			return 1;
////		}
////	}
////
////	// Go to the link face of e1.
////	point2tetorg(e1, searchtet, point2tet, tetlist);
////	enextesymself(searchtet);
////
////	std::vector<tethandle> recordtetlist; // recorded tet list
////
////										  // Search e2.
////	for (i = 0; i < 3; i++) {
////		pt = apex(searchtet, tetlist);
////		if (pt == e2) {
////			// Found. 'searchtet' is [#,#,e2,e1].
////			eorgoppo(searchtet, *tedge); // [e1,e2,#,#].
////			return 1;
////		}
////		enextself(searchtet);
////	}
////
////	// Get the adjacent link face at 'searchtet'.
////	fnext(searchtet, neightet, neighborlist);
////	esymself(neightet);
////	// assert(oppo(neightet) == e1);
////	pt = apex(neightet, tetlist);
////	if (pt == e2) {
////		// Found. 'neightet' is [#,#,e2,e1].
////		eorgoppo(neightet, *tedge); // [e1,e2,#,#].
////		return 1;
////	}
////
////	// Continue searching in the link face of e1.
////	markerlist[searchtet.id] = 1; // initial value of markerlist must be 0
////	recordtetlist.push_back(searchtet);
////	markerlist[neightet.id] = 1;
////	recordtetlist.push_back(neightet);
////
////	done = 0;
////
////	for (i = 0; (i < recordtetlist.size()) && !done; i++) {
////		parytet = recordtetlist[i];
////		searchtet = parytet;
////		for (j = 0; (j < 2) && !done; j++) {
////			enextself(searchtet);
////			fnext(searchtet, neightet, neighborlist);
////			if (!markerlist[neightet.id]) {
////				esymself(neightet);
////				pt = apex(neightet, tetlist);
////				if (pt == e2) {
////					// Found. 'neightet' is [#,#,e2,e1].
////					eorgoppo(neightet, *tedge);
////					done = 1;
////				}
////				else {
////					markerlist[neightet.id] = 1;
////					recordtetlist.push_back(neightet);
////				}
////			}
////		} // j
////	} // i 
////
////	  // Uninfect the list of visited tets.
////	for (i = 0; i < recordtetlist.size(); i++) {
////		parytet = recordtetlist[i];
////		markerlist[parytet.id] = 0;
////	}
////
////	return done;
////}

/* Refinement */

// Insert point
__global__ void kernelCheckAbortiveElements(
	int* d_insertidxlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int insertiontype,
	int numofinsertpt
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofinsertpt)
		return;

	int insertid = d_insertidxlist[pos];
	bool flag;
	if (insertiontype == 0)
		flag = d_segstatus[insertid].isAbortive();
	else if (insertiontype == 1)
		flag = d_tristatus[insertid].isAbortive();
	else if (insertiontype == 2)
		flag = d_tetstatus[insertid].isAbortive();

	if (flag)
		d_threadmarker[pos] = -1;
}

__global__ void kernelCheckInsertRadius_Seg(
	int* d_segidlist,
	REAL* d_pointlist,
	REAL* d_pointradius,
	int* d_seglist,
	tristatus* d_segstatus,
	int* d_segencmarker,
	int* d_threadmarker,
	int numofseg
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofseg)
		return;

	int segId = d_segidlist[pos];
	if (d_segstatus[segId].isAbortive())
	{
		d_threadmarker[pos] = -1;
		return;
	}

	int encptidx = d_segencmarker[pos];
	if (encptidx != MAXINT) // not encroached by splitting segment and subface routines
		return;

	trihandle splitseg(segId, 0);
	int ipa, ipb;
	ipa = cudamesh_sorg(splitseg, d_seglist);
	ipb = cudamesh_sdest(splitseg, d_seglist);
	REAL *pa, *pb;
	pa = cudamesh_id2pointlist(ipa, d_pointlist);
	pb = cudamesh_id2pointlist(ipb, d_pointlist);
	REAL len = cudamesh_distance(pa, pb);
	REAL smrrv = d_pointradius[ipa];
	REAL rrv = d_pointradius[ipb];
	if (rrv > 0)
	{
		if (smrrv > 0)
		{
			if (rrv < smrrv)
			{
				smrrv = rrv;
			}
		}
		else
		{
			smrrv = rrv;
		}
	}
	if (smrrv > 0)
	{
		if ((fabs(smrrv - len) / len) < EPSILON)
			smrrv = len;
		if (len < smrrv)
		{
			d_segstatus[segId].setAbortive(true);
			d_threadmarker[pos] = -1;
			return;
		}
	}
}

__global__ void kernelComputePriority_Seg(
	int* d_segidlist,
	int* d_threadlist,
	int* d_seglist,
	REAL* d_pointlist,
	int* d_priority,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int segId = d_segidlist[threadId];
	trihandle splitseg(segId, 0);
	int ipa, ipb;
	ipa = cudamesh_sorg(splitseg, d_seglist);
	ipb = cudamesh_sdest(splitseg, d_seglist);
	REAL *pa, *pb;
	pa = cudamesh_id2pointlist(ipa, d_pointlist);
	pb = cudamesh_id2pointlist(ipb, d_pointlist);

	REAL len = cudamesh_distance(pa, pb);
	d_priority[threadId] = __float_as_int((float)(1/len));
}

__global__ void kernelInitSearchTet_Seg(
	int* d_segidlist,
	int* d_threadlist,
	tethandle* d_seg2tetlist,
	tethandle* d_searchtetlist,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int segId = d_segidlist[threadId];

	trihandle splitseg(segId, 0);
	tethandle searchtet;
	cudamesh_sstpivot1(splitseg, searchtet, d_seg2tetlist);
	d_searchtetlist[threadId] = searchtet;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int subfaceid = d_subfaceidlist[pos];
	if (d_tristatus[subfaceid].isAbortive())
	{
		d_threadmarker[pos] = -1;
		return;
	}

	int encptidx = d_subfaceencmarker[subfaceid];
	if (encptidx == MAXINT) // Mark as encroached when trying to split a tet
		return;

	trihandle parentseg, parentsh;
	trihandle splitfac(subfaceid, 0);
	REAL rv, rp;
	REAL* newpt = d_insertptlist + 3 * pos;
	REAL* encpt = cudamesh_id2pointlist(encptidx, d_pointlist);

	rv = cudamesh_distance(newpt, encpt);
	if (d_pointtypelist[encptidx] == FREESEGVERTEX)
	{
		parentseg = d_point2trilist[encptidx];
		if (cudamesh_segfacetadjacent(parentseg.id, splitfac.id,
			d_seg2parentidxlist, d_segparentendpointidxlist, 
			d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
		{
			//printf("Adjacent: Seg #%d, Subface #%d\n",
			//	d_seg2parentidxlist[parentseg.id], d_tri2parentidxlist[splitfac.id]);
			rp = d_pointradius[encptidx];
			if (rv < (sqrt(2.0) * rp))
			{
				// This insertion may cause no termination.
				d_threadmarker[pos] = -1; // Reject the insertion of newpt.
				d_tristatus[subfaceid].setAbortive(true);
			}
		}
	}
	else if (d_pointtypelist[encptidx] == FREEFACETVERTEX)
	{
		parentsh = d_point2trilist[encptidx];
		if (cudamesh_facetfacetadjacent(parentsh.id, splitfac.id,
			d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
		{
			//printf("Adjacent: Subface #%d, Subface #%d\n",
			//	d_tri2parentidxlist[parentsh.id], d_tri2parentidxlist[splitfac.id]);
			rp = d_pointradius[encptidx];
			if (rv < rp)
			{
				d_threadmarker[pos] = -1; // Reject the insertion of newpt.
				d_tristatus[subfaceid].setAbortive(true);
			}
		}
	}
}

__global__ void kernelInitSearchshList(
	int* d_subfaceidlist,
	int* d_threadlist,
	trihandle* d_searchsh,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int subfaceid = d_subfaceidlist[threadId];
	d_searchsh[threadId] = trihandle(subfaceid, 0);
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];

	trihandle neighsh;
	trihandle *searchsh = d_searchsh + threadId;
	REAL *searchpt = d_insertptlist + 3 * threadId;
	REAL *pa, *pb, *pc;
	unsigned long *randomseed = d_randomseed + pos;
	REAL abvpt[3];

	enum locateresult loc;
	enum { MOVE_BC, MOVE_CA } nextmove;
	REAL ori, ori_bc, ori_ca;
	int i;

	pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
	pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
	pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

	// Calculate an above point for this facet.
	cudamesh_calculateabovepoint4(searchpt, pa, pb, pc, abvpt);

	// 'abvpt' is given. Make sure it is above [a,b,c]
	ori = cuda_orient3d(pa, pb, pc, abvpt);
	assert(ori != 0); // SELF_CHECK
	if (ori > 0) {
		cudamesh_sesymself(*searchsh); // Reverse the face orientation.
	}

	// Find an edge of the face s.t. p lies on its right-hand side (CCW).
	for (i = 0; i < 3; i++) {
		pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
		ori = cuda_orient3d(pa, pb, abvpt, searchpt);
		if (ori > 0) break;
		cudamesh_senextself(*searchsh);
	}
	assert(i < 3); // SELF_CHECK

	pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

	if (pc[0] == searchpt[0] && pc[1] == searchpt[1] && pc[2] == searchpt[2]) {
		cudamesh_senext2self(*searchsh);
		loc = ONVERTEX;
	}
	else
	{
		while (1) {
			ori_bc = cuda_orient3d(pb, pc, abvpt, searchpt);
			ori_ca = cuda_orient3d(pc, pa, abvpt, searchpt);

			if (ori_bc < 0) {
				if (ori_ca < 0) { // (--)
								  // Any of the edges is a viable move.
					if (cudamesh_randomnation(randomseed,2)) {
						nextmove = MOVE_CA;
					}
					else {
						nextmove = MOVE_BC;
					}
				}
				else { // (-#)
					   // Edge [b, c] is viable.
					nextmove = MOVE_BC;
				}
			}
			else {
				if (ori_ca < 0) { // (#-)
								  // Edge [c, a] is viable.
					nextmove = MOVE_CA;
				}
				else {
					if (ori_bc > 0) {
						if (ori_ca > 0) { // (++)
							loc = ONFACE;  // Inside [a, b, c].
							break;
						}
						else { // (+0)
							cudamesh_senext2self(*searchsh); // On edge [c, a].
							loc = ONEDGE;
							break;
						}
					}
					else { // ori_bc == 0
						if (ori_ca > 0) { // (0+)
							cudamesh_senextself(*searchsh); // On edge [b, c].
							loc = ONEDGE;
							break;
						}
						else { // (00)
							   // p is coincident with vertex c. 
							cudamesh_senext2self(*searchsh);
							loc = ONVERTEX;
							break;
						}
					}
				}
			}

			// Move to the next face.
			if (nextmove == MOVE_BC) {
				cudamesh_senextself(*searchsh);
			}
			else {
				cudamesh_senext2self(*searchsh);
			}

			// NON-convex case. Check if we will cross a boundary.
			if (cudamesh_isshsubseg(*searchsh, d_tri2seglist)) {
				loc = ENCSEGMENT;
				break;
			}

			cudamesh_spivot(*searchsh, neighsh, d_tri2trilist);
			if (neighsh.id == -1) {
				loc =  OUTSIDE; // A hull edge.
				break;
			}

			// Adjust the edge orientation.
			if (cudamesh_sorg(neighsh, d_trifacelist) != cudamesh_sdest(*searchsh, d_trifacelist)) {
				cudamesh_sesymself(neighsh);
			}
			assert(cudamesh_sorg(neighsh, d_trifacelist) == cudamesh_sdest(*searchsh, d_trifacelist)); // SELF_CHECK

			// Update the newly discovered face and its endpoints.
			*searchsh = neighsh;
			pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
			pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
			pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

			if (pc == searchpt) {
				cudamesh_senext2self(*searchsh);
				loc = ONVERTEX;
				break;
			}

		} // while (1)
	}

	d_pointlocation[threadId] = loc;
	if (!(loc == ONFACE || loc == ONEDGE))
	{
		int subfaceid = d_subfaceidlist[threadId];
		d_tristatus[subfaceid].setAbortive(true); // mark the encroached subface rather than the located one
		d_threadmarker[threadId] = -1;
		return;
	}

	tethandle searchtet;
	cudamesh_stpivot(*searchsh, searchtet, d_tri2tetlist);
	d_searchtetlist[threadId] = searchtet;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];

	int subfaceid = d_insertidxlist[threadId];

	trihandle splitsh = trihandle(subfaceid, 0);

	int ipa, ipb, ipc;
	ipa = cudamesh_sorg(splitsh, d_trifacelist);
	ipb = cudamesh_sdest(splitsh, d_trifacelist);
	ipc = cudamesh_sapex(splitsh, d_trifacelist);
	
	REAL *pa, *pb, *pc;
	pa = cudamesh_id2pointlist(ipa, d_pointlist);
	pb = cudamesh_id2pointlist(ipb, d_pointlist);
	pc = cudamesh_id2pointlist(ipc, d_pointlist);

	// Compute the area of this 3D triangle
	REAL AB[3], AC[3];
	int i;
	for (i = 0; i < 3; i++)
	{
		AB[i] = pb[i] - pa[i];
		AC[i] = pc[i] - pa[i];
	}

	REAL area =
		sqrt((AB[1] * AC[2] - AB[2] * AC[1])*(AB[1] * AC[2] - AB[2] * AC[1]) +
		 (AB[2] * AC[0] - AB[0] * AC[2])*(AB[2] * AC[0] - AB[0] * AC[2]) +
		 (AB[0] * AC[1] - AB[1] * AC[0])*(AB[0] * AC[1] - AB[1] * AC[0])) / 2;

	d_priority[threadId] = __float_as_int((float)(1/ area));

	//int offsetid = d_tri2parentidxlist[subfaceid];
	//REAL* pt[4];
	//for (int i = d_triid2parentoffsetlist[offsetid]; i < d_triid2parentoffsetlist[offsetid]; i++)
	//{
	//	
	//}
}

__global__ void kernelCheckInsertRadius_Tet(
	int* d_tetidlist,
	REAL* d_pointlist,
	REAL* d_pointradius,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetid = d_tetidlist[pos];
	if (d_tetstatus[tetid].isAbortive())
	{
		d_threadmarker[pos] = -1;
		return;
	}

	tethandle chktet(tetid, 11), checkedge;
	int ie1, ie2;
	int i, j;
	REAL *e1, *e2;
	REAL smlen = 0;
	REAL rrv, smrrv;
	REAL elen[6];

	// Get the shortest edge of this tet.
	checkedge.id = chktet.id;
	for (i = 0; i < 6; i++) {
		checkedge.ver = raw_edge2ver[i];
		ie1 = cudamesh_org(checkedge, d_tetlist);
		ie2 = cudamesh_dest(checkedge, d_tetlist);
		e1 = cudamesh_id2pointlist(ie1, d_pointlist);
		e2 = cudamesh_id2pointlist(ie2, d_pointlist);
		elen[i] = cudamesh_distance(e1, e2);
		if (i == 0) {
			smlen = elen[i];
			j = 0;
		}
		else {
			if (elen[i] < smlen) {
				smlen = elen[i];
				j = i;
			}
		}
	}
	// Check if the edge is too short.
	checkedge.ver = raw_edge2ver[j];
	// Get the smallest rrv of e1 and e2.
	// Note: if rrv of e1 and e2 is zero. Do not use it.
	ie1 = cudamesh_org(checkedge, d_tetlist);
	smrrv = d_pointradius[ie1];
	ie2 = cudamesh_dest(checkedge, d_tetlist);
	rrv = d_pointradius[ie2];
	if (rrv > 0) {
		if (smrrv > 0) {
			if (rrv < smrrv) {
				smrrv = rrv;
			}
		}
		else {
			smrrv = rrv;
		}
	}
	if (smrrv > 0) {
		// To avoid rounding error, round smrrv before doing comparison.
		if ((fabs(smrrv - smlen) / smlen) <EPSILON) {
			smrrv = smlen;
		}
		if (smrrv > smlen) {
			d_tetstatus[tetid].setAbortive(true);
			d_threadmarker[pos] = -1;
			return;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	int tetid = d_tetidlist[threadId];
	tethandle* searchtet = d_searchtetlist + threadId;
	REAL* searchpt = d_insertptlist + 3 * threadId;
	unsigned long* randomseed = d_randomseed + pos;

	REAL *torg, *tdest, *tapex, *toppo;
	enum { ORGMOVE, DESTMOVE, APEXMOVE } nextmove;
	REAL ori, oriorg, oridest, oriapex;
	enum locateresult loc = OUTSIDE;
	int t1ver;
	int s;
	int step = 1;

	// Init searchtet
	searchtet->id = tetid;
	searchtet->ver = 11;

	// Check if we are in the outside of the convex hull.
	if (cudamesh_ishulltet(*searchtet, d_tetlist)) {
		// Get its adjacent tet (inside the hull).
		searchtet->ver = 3;
		cudamesh_fsymself(*searchtet, d_neighborlist);
	}

	// Let searchtet be the face such that 'searchpt' lies above to it.
	for (searchtet->ver = 0; searchtet->ver < 4; searchtet->ver++) {
		torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
		tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
		tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);
		ori = cuda_orient3d(torg, tdest, tapex, searchpt);
		if (ori < 0.0) break;
	}
	assert(searchtet->ver != 4);

	// Walk through tetrahedra to locate the point.
	while (true) {

		toppo = cudamesh_id2pointlist(cudamesh_oppo(*searchtet, d_tetlist), d_pointlist);

		// Check if the vertex is we seek.
		if (toppo[0] == searchpt[0] && toppo[1] == searchpt[1] && toppo[2] == searchpt[2]) {
			// Adjust the origin of searchtet to be searchpt.
			cudamesh_esymself(*searchtet);
			cudamesh_eprevself(*searchtet);
			loc = ONVERTEX; // return ONVERTEX;
			break;
		}

		// We enter from one of serarchtet's faces, which face do we exit?
		oriorg = cuda_orient3d(tdest, tapex, toppo, searchpt);
		oridest = cuda_orient3d(tapex, torg, toppo, searchpt);
		oriapex = cuda_orient3d(torg, tdest, toppo, searchpt);

		// Now decide which face to move. It is possible there are more than one
		//   faces are viable moves. If so, randomly choose one.
		if (oriorg < 0) {
			if (oridest < 0) {
				if (oriapex < 0) {
					// All three faces are possible.
					s = cudamesh_randomnation(randomseed, 3); // 's' is in {0,1,2}.
					if (s == 0) {
						nextmove = ORGMOVE;
					}
					else if (s == 1) {
						nextmove = DESTMOVE;
					}
					else {
						nextmove = APEXMOVE;
					}
				}
				else {
					// Two faces, opposite to origin and destination, are viable.
					//s = randomnation(2); // 's' is in {0,1}.
					if (cudamesh_randomnation(randomseed, 2)) {
						nextmove = ORGMOVE;
					}
					else {
						nextmove = DESTMOVE;
					}
				}
			}
			else {
				if (oriapex < 0) {
					// Two faces, opposite to origin and apex, are viable.
					//s = randomnation(2); // 's' is in {0,1}.
					if (cudamesh_randomnation(randomseed, 2)) {
						nextmove = ORGMOVE;
					}
					else {
						nextmove = APEXMOVE;
					}
				}
				else {
					// Only the face opposite to origin is viable.
					nextmove = ORGMOVE;
				}
			}
		}
		else {
			if (oridest < 0) {
				if (oriapex < 0) {
					// Two faces, opposite to destination and apex, are viable.
					//s = randomnation(2); // 's' is in {0,1}.
					if (cudamesh_randomnation(randomseed, 2)) {
						nextmove = DESTMOVE;
					}
					else {
						nextmove = APEXMOVE;
					}
				}
				else {
					// Only the face opposite to destination is viable.
					nextmove = DESTMOVE;
				}
			}
			else {
				if (oriapex < 0) {
					// Only the face opposite to apex is viable.
					nextmove = APEXMOVE;
				}
				else {
					// The point we seek must be on the boundary of or inside this
					//   tetrahedron. Check for boundary cases.
					if (oriorg == 0) {
						// Go to the face opposite to origin.
						cudamesh_enextesymself(*searchtet);
						if (oridest == 0) {
							cudamesh_eprevself(*searchtet); // edge oppo->apex
							if (oriapex == 0) {
								// oppo is duplicated with p.
								loc = ONVERTEX; // return ONVERTEX;
								break;
							}
							loc = ONEDGE; // return ONEDGE;
							break;
						}
						if (oriapex == 0) {
							cudamesh_enextself(*searchtet); // edge dest->oppo
							loc = ONEDGE; // return ONEDGE;
							break;
						}
						loc = ONFACE; // return ONFACE;
						break;
					}
					if (oridest == 0) {
						// Go to the face opposite to destination.
						cudamesh_eprevesymself(*searchtet);
						if (oriapex == 0) {
							cudamesh_eprevself(*searchtet); // edge oppo->org
							loc = ONEDGE; // return ONEDGE;
							break;
						}
						loc = ONFACE; // return ONFACE;
						break;
					}
					if (oriapex == 0) {
						// Go to the face opposite to apex
						cudamesh_esymself(*searchtet);
						loc = ONFACE; // return ONFACE;
						break;
					}
					loc = INTETRAHEDRON; // return INTETRAHEDRON;
					break;
				}
			}
		}

		// Move to the selected face.
		if (nextmove == ORGMOVE) {
			cudamesh_enextesymself(*searchtet);
		}
		else if (nextmove == DESTMOVE) {
			cudamesh_eprevesymself(*searchtet);
		}
		else {
			cudamesh_esymself(*searchtet);
		}
		// Move to the adjacent tetrahedron (maybe a hull tetrahedron).
		cudamesh_fsymself(*searchtet, d_neighborlist);
		if (cudamesh_oppo(*searchtet, d_tetlist) == -1) {
			loc = OUTSIDE; // return OUTSIDE;
			break;
		}

		// Retreat the three vertices of the base face.
		torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
		tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
		tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);

		step++;

	} // while (true)

	d_pointlocation[threadId] = loc;

	// set weighted priority
	//REAL vol = cudamesh_tetrahedronvolume(tetid, d_pointlist, d_tetlist);
	//REAL wp = 0.5*vol + 0.5*step;
	//d_priority[threadId] = __float_as_int((float)(1 / wp));

	if (loc == ONVERTEX)
	{
		d_tetstatus[tetid].setAbortive(true);
		d_threadmarker[threadId] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// initial cavity
	// mark all tets share at this edge
	int count = 0, i;
	int old;
	uint64 marker, oldmarker;
	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId);

	if (loc == ONEDGE)
	{
		spintet = searchtet;
		while (1) {
			// check if already lost
			if (d_threadmarker[threadId] == -1) // lost because of other threads
			{
				count = 0;
				break;
			}

			// marking competition
			oldmarker = atomicMin(d_tetmarker + spintet.id, marker);
			if (marker < oldmarker) // winned
			{
				old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != MAXUINT)
				{
					d_threadmarker[old] = -1;
					atomicMin(d_initialcavitysize + old, 0);
					atomicMin(d_initialsubcavitysize + old, 0);
				}
			}
			else // lost
			{
				d_threadmarker[threadId] = -1;
				count = 0;
				break;
			}

			count++;
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id) break;
		} // while (1)
	}
	else if (loc == ONFACE)
	{
		// check if already lost
		if (d_threadmarker[threadId] == -1) // lost because of other threads
		{
			count = 0;
		}
		else // mark two adjacent tets on the face 
		{
			spintet = searchtet;
			for (i = 0; i < 2; i++)
			{
				// marking competition
				oldmarker = atomicMin(d_tetmarker + spintet.id, marker);
				if (marker < oldmarker) // winned
				{
					old = cudamesh_getUInt64PriorityIndex(oldmarker);
					if (old != MAXUINT)
					{
						d_threadmarker[old] = -1;
						atomicMin(d_initialcavitysize + old, 0);
						atomicMin(d_initialsubcavitysize + old, 0);
					}
				}
				else // lost
				{
					d_threadmarker[threadId] = -1;
					count = 0;
					break;
				}
				count++;
				spintet = d_neighborlist[4 * searchtet.id + (searchtet.ver & 3)];
			}
		}
	}
	else if (loc == INTETRAHEDRON || loc == OUTSIDE)
	{
		// check if already lost
		if (d_threadmarker[threadId] == -1) // lost because of other threads
		{
			count = 0;
		}
		else // mark four adjecent tets
		{
			// marking competition
			oldmarker = atomicMin(d_tetmarker + searchtet.id, marker);
			if (marker < oldmarker) // winned
			{
				count = 1;
				old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != MAXUINT)
				{
					d_threadmarker[old] = -1;
					atomicMin(d_initialcavitysize + old, 0);
					atomicMin(d_initialsubcavitysize + old, 0);
				}
			}
			else // lost
			{
				d_threadmarker[threadId] = -1;
				count = 0;
			}
		}
	}

	atomicMin(d_initialcavitysize + threadId, count);

	// Initial subcavity
	// Count all subfaces share at this edge.
	int scount = 0;
	if (count == 0)
		scount = 0;
	else
	{
		trihandle splitsh;
		if (loc == ONEDGE)
		{
			if (threadmarker == 0)
			{
				int segId = d_insertidxlist[threadId];
				trihandle splitseg(segId, 0);
				atomicMin(d_segmarker + splitseg.id, threadId);
				cudamesh_spivot(splitseg, splitsh, d_seg2trilist);
			}
			else if (threadmarker == 1)
			{
				splitsh = d_searchsh[threadId];
			}

			if (splitsh.id != -1)
			{
				int pa = cudamesh_sorg(splitsh, d_trifacelist);
				trihandle neighsh = splitsh;
				while (1) {
					// Check if already lost
					if (d_threadmarker[threadId] == -1) // lost because of other threads
					{
						scount = 0;
						break;
					}

					// Adjust the origin of its edge to be 'pa'.
					if (cudamesh_sorg(neighsh, d_trifacelist) != pa) {
						cudamesh_sesymself(neighsh);
					}

					// Mark this face
					atomicMin(d_trimarker + neighsh.id, marker);

					// count this face
					scount++;

					// Go to the next face at the edge.
					cudamesh_spivotself(neighsh, d_tri2trilist);
					// Stop if all faces at the edge have been visited.
					if (neighsh.id == splitsh.id) break;
					if (neighsh.id == -1) break;
				} // while (1)
			}
		}
		else if (loc == ONFACE)
		{
			if (threadmarker == 1)
			{
				// Check if already lost
				if (d_threadmarker[threadId] == -1) // lost because of other threads
				{
					scount = 0;
				}
				else
				{
					splitsh = d_searchsh[threadId];

					// Mark this face
					atomicMin(d_trimarker + splitsh.id, marker);

					// count this face
					scount++;
				}
			}
		}
	}

	atomicMin(d_initialsubcavitysize + threadId, scount);
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	bool reusecavity = false;
	int eleidx = d_insertidxlist[threadId];
	if (threadmarker == 0)
	{
		if (d_segstatus[eleidx].isCavityReuse())
		{
			atomicMin(d_initialcavitysize + threadId, 0);
			reusecavity = true;
		}
	}
	else if (threadmarker == 1)
	{
		if (d_tristatus[eleidx].isCavityReuse())
		{
			atomicMin(d_initialcavitysize + threadId, 0);
			reusecavity = true;
		}
	}
	else if (threadmarker == 2)
	{
		if (d_tetstatus[eleidx].isCavityReuse())
		{
			atomicMin(d_initialcavitysize + threadId, 0);
			reusecavity = true;
		}
	}

	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// initial cavity
	// mark all tets share at this edge
	int count = 0, i;
	int old;
	uint64 marker, oldmarker;
	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);

	if (!reusecavity)
	{
		if (loc == ONEDGE)
		{
			spintet = searchtet;
			while (1) {
				// check if already lost
				if (d_threadmarker[threadId] == -1) // lost because of other threads
				{
					count = 0;
					break;
				}

				// marking competition
				oldmarker = atomicMax(d_tetmarker + spintet.id, marker);
				if (marker > oldmarker) // winned
				{
					old = cudamesh_getUInt64PriorityIndex(oldmarker);
					if (old != 0) // marked by others
					{
						d_threadmarker[old - 1] = -1;
						atomicMin(d_initialcavitysize + old - 1, 0);
						atomicMin(d_initialsubcavitysize + old - 1, 0);
					}
				}
				else // lost
				{
					d_threadmarker[threadId] = -1;
					count = 0;
					break;
				}

				count++;
				cudamesh_fnextself(spintet, d_neighborlist);
				if (spintet.id == searchtet.id) break;
			} // while (1)
		}
		else if (loc == ONFACE)
		{
			// check if already lost
			if (d_threadmarker[threadId] == -1) // lost because of other threads
			{
				count = 0;
			}
			else // mark two adjacent tets on the face 
			{
				spintet = searchtet;
				for (i = 0; i < 2; i++)
				{
					// marking competition
					oldmarker = atomicMax(d_tetmarker + spintet.id, marker);
					if (marker > oldmarker) // winned
					{
						old = cudamesh_getUInt64PriorityIndex(oldmarker);
						if (old != 0)
						{
							d_threadmarker[old - 1] = -1;
							atomicMin(d_initialcavitysize + old - 1, 0);
							atomicMin(d_initialsubcavitysize + old - 1, 0);
						}
					}
					else // lost
					{
						d_threadmarker[threadId] = -1;
						count = 0;
						break;
					}
					count++;
					spintet = d_neighborlist[4 * searchtet.id + (searchtet.ver & 3)];
				}
			}
		}
		else if (loc == INTETRAHEDRON || loc == OUTSIDE)
		{
			// check if already lost
			if (d_threadmarker[threadId] == -1) // lost because of other threads
			{
				count = 0;
			}
			else // mark four adjecent tets
			{
				// marking competition
				oldmarker = atomicMax(d_tetmarker + searchtet.id, marker);
				if (marker > oldmarker) // winned
				{
					count = 1;
					old = cudamesh_getUInt64PriorityIndex(oldmarker);
					if (old != 0)
					{
						d_threadmarker[old - 1] = -1;
						atomicMin(d_initialcavitysize + old - 1, 0);
						atomicMin(d_initialsubcavitysize + old - 1, 0);
					}
				}
				else // lost
				{
					d_threadmarker[threadId] = -1;
					count = 0;
				}
			}
		}
	}

	atomicMin(d_initialcavitysize + threadId, count);

	// Initial subcavity
	// Count all subfaces share at this edge.
	int scount = 0;
	if (count == 0 && !reusecavity)
		scount = 0;
	else
	{
		trihandle splitsh;
		if (loc == ONEDGE)
		{
			if (threadmarker == 0)
			{
				int segId = d_insertidxlist[threadId];
				trihandle splitseg(segId, 0);
				atomicMax(d_segmarker + splitseg.id, threadId + 1);
				cudamesh_spivot(splitseg, splitsh, d_seg2trilist);
			}
			else if (threadmarker == 1)
			{
				splitsh = d_searchsh[threadId];
			}

			if (splitsh.id != -1)
			{
				int pa = cudamesh_sorg(splitsh, d_trifacelist);
				trihandle neighsh = splitsh;
				while (1) {
					// Check if already lost
					if (d_threadmarker[threadId] == -1) // lost because of other threads
					{
						scount = 0;
						break;
					}

					// Adjust the origin of its edge to be 'pa'.
					if (cudamesh_sorg(neighsh, d_trifacelist) != pa) {
						cudamesh_sesymself(neighsh);
					}

					// Mark this face
					atomicMax(d_trimarker + neighsh.id, marker);

					// count this face
					scount++;

					// Go to the next face at the edge.
					cudamesh_spivotself(neighsh, d_tri2trilist);
					// Stop if all faces at the edge have been visited.
					if (neighsh.id == splitsh.id) break;
					if (neighsh.id == -1) break;
				} // while (1)
			}
		}
		else if (loc == ONFACE)
		{
			if (threadmarker == 1)
			{
				// Check if already lost
				if (d_threadmarker[threadId] == -1) // lost because of other threads
				{
					scount = 0;
				}
				else
				{
					splitsh = d_searchsh[threadId];

					// Mark this face
					atomicMax(d_trimarker + splitsh.id, marker);

					// count this face
					scount++;
				}
			}
		}
	}

	atomicMin(d_initialsubcavitysize + threadId, scount);
}

__device__ int elementId2threadId(
	int* d_insertidxlist,
	int eleidx,
	int eletype,
	int numofencsubseg,
	int numofencsubface,
	int numofbadelement
)
{
	int low, high;

	if (eletype == 0) // subseg
	{
		low = 0;
		high = numofencsubseg - 1;
	}
	else if (eletype == 1) // subface
	{
		low = numofencsubseg;
		high = numofencsubface - 1;
	}
	else if (eletype == 2) // tet
	{
		low = numofencsubface;
		high = numofbadelement - 1;
	}

	int middle, val;
	while (high >= low)
	{
		middle = (low + high) / 2;
		val = d_insertidxlist[middle];
		if (val == eleidx)
			return middle;
		else if (val < eleidx)
			low = middle + 1;
		else
			high = middle - 1;
	}

	return -1;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	tethandle checktet = d_recordoldtetlist[pos];
	int eleidx = d_recordoldtetidx[pos];

	// use binary search to find its threadId
	int threadId = elementId2threadId(d_insertidxlist,
		eleidx, checktet.ver, numofencsubseg, numofencsubface, numofbadelement);

	// check if the element is deleted or not
	if (checktet.ver == 0) // subseg
	{
		if (d_segstatus[eleidx].isEmpty() || !d_segstatus[eleidx].isCavityReuse())
		{
			d_recordoldtetidx[pos] = -1;
			if (threadId != -1)
				d_threadmarker[threadId] = -1;
			return;
		}
	}
	else if (checktet.ver == 1) // subface
	{
		if (d_tristatus[eleidx].isEmpty() || !d_tristatus[eleidx].isCavityReuse())
		{
			d_recordoldtetidx[pos] = -1;
			if (threadId != -1)
				d_threadmarker[threadId] = -1;
			return;
		}
	}
	else if (checktet.ver == 2) // tet
	{
		if (d_tetstatus[eleidx].isEmpty() || !d_tetstatus[eleidx].isCavityReuse())
		{
			d_recordoldtetidx[pos] = -1;
			if (threadId != -1)
				d_threadmarker[threadId] = -1;
			return;
		}
	}

	// check if the tetrahedron is deleted or not
	if (d_tetstatus[checktet.id].isEmpty())
	{
		d_recordoldtetidx[pos] = -1;
		return;
	}

	if (threadId == -1) // this element is not going to be inserted in this iteration
		return;

	if (d_threadmarker[threadId] < 0) // lost already
		return;

	// this element is going to be inserted, check if all tetrahedra are still in its cavity
	REAL* insertpt = d_insertptlist + 3 * threadId;
	int old;
	uint64 marker, oldmarker;
	// here we use threadId + 1, 0 thread index is reserved for default marker
	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);

	bool enqflag = false;
	double sign;
	// Get four endpoints of cavetet
	REAL *pts[4], wts[4];
	int idx[4];
	for (int i = 0; i < 4; i++)
	{
		idx[i] = d_tetlist[4 * checktet.id + i];
		if (idx[i] != -1)
		{
			pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
		}
		else
			pts[i] = NULL;
	}
	// Test if cavetet is included in the (enlarged) cavity
	if (idx[3] != -1)
	{
		sign = cudamesh_insphere_s(pts[0], pts[1], pts[2], pts[3], insertpt,
			idx[0], idx[1], idx[2], idx[3], MAXINT);
		enqflag = (sign < 0.0);
	}
	else // a hull tet
	{
		// We FIRST finclude it in the initial cavity if its adjacent tet is
		// not Delaunay wrt p. Will validate it later on.
		tethandle neineitet = d_neighborlist[4 * checktet.id + 3];
		if (d_tetmarker[neineitet.id] != marker) // need to check
		{
			// Get four endpoints of neineitet
			for (int i = 0; i < 4; i++)
			{
				idx[i] = d_tetlist[4 * neineitet.id + i];
				if (idx[i] != -1)
					pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
				else
					pts[i] = NULL;
			}
			if (idx[3] == -1)
			{
				enqflag = false;
			}
			else
			{
				sign = cudamesh_insphere_s(pts[0], pts[1], pts[2], pts[3], insertpt,
					idx[0], idx[1], idx[2], idx[3], MAXINT);
				enqflag = (sign < 0.0);
			}
		}
		else
		{
			enqflag = true;
		}
	}
	// Count size
	if (enqflag)
	{
		oldmarker = atomicMax(d_tetmarker + checktet.id, marker);
		if (marker > oldmarker) // I winned
		{
			old = cudamesh_getUInt64PriorityIndex(oldmarker);
			if (old != 0)
			{
				d_threadmarker[old - 1] = -1;
				d_initialcavitysize[old - 1] = 0;
				d_initialsubcavitysize[old - 1] = 0;
			}
		}
		else if (marker < oldmarker) // I lost
		{
			d_threadmarker[threadId] = -1;
			d_initialsubcavitysize[threadId] = 0;
		}
		d_recordoldtetidx[pos] = -(threadId + 2);
	}
	else
	{
		d_recordoldtetidx[pos] = -1;
	}
}

__global__ void kernelKeepRecordOldtet(
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_recordoldtetidx[pos];
	if (threadId >= -1) // invalid or do not participate
		return;

	threadId = -threadId - 2;
	if (d_threadmarker[threadId] < 0)
	{
		d_recordoldtetidx[pos] = d_insertidxlist[threadId];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// cavities
	int icsindex = d_initialcavityindices[threadId];
	int count = 0;
	int prev = -1;
	int icavityIdx;
	int tetidxfactor = 4;

	if (loc == ONEDGE)
	{
		spintet = searchtet;
		while (1) {
			// initial cavity index
			icavityIdx = icsindex + count;

			// add to tet list
			cudamesh_eorgoppo(spintet, neightet);
			neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavetetlist[tetidxfactor * icavityIdx] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx] = (prev == -1) ? -1 : tetidxfactor * prev + 1;
			d_cavetetnext[tetidxfactor * icavityIdx] = tetidxfactor * icavityIdx + 1;
			if (prev != -1)
				d_cavetetnext[tetidxfactor * prev + 1] = tetidxfactor * icavityIdx;
			d_cavethreadidx[tetidxfactor * icavityIdx] = threadId;

			cudamesh_edestoppo(spintet, neightet);
			neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavetetlist[tetidxfactor * icavityIdx + 1] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + 1] = tetidxfactor * icavityIdx;
			d_cavetetnext[tetidxfactor * icavityIdx + 1] = -1;
			d_cavethreadidx[tetidxfactor * icavityIdx + 1] = threadId;

			// add to old tet list
			d_caveoldtetlist[icavityIdx] = spintet; // current tet
			d_caveoldtetprev[icavityIdx] = prev; // previous
			d_caveoldtetnext[icavityIdx] = -1; // next, set to -1 first
			if (prev != -1)
				d_caveoldtetnext[prev] = icavityIdx; // previous next, set to me

			if (count == 0)
			{
				d_caveoldtethead[threadId] = icavityIdx;
				d_cavetethead[threadId] = tetidxfactor * icavityIdx;
			}

			// next iteration
			prev = icavityIdx;
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id)
			{
				d_caveoldtettail[threadId] = icavityIdx;
				d_cavetettail[threadId] = tetidxfactor * icavityIdx + 1;
				break;
			}
			count++;
		} // while (1)
	}
	else if (loc == ONFACE)
	{
		int i, j;
		// initial cavity index
		icavityIdx = icsindex;

		// add to tet and old tet list
		j = (searchtet.ver & 3);
		for (i = 1; i < 4; i++)
		{
			neightet = d_neighborlist[4 * searchtet.id + (j + i) % 4];
			d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + i - 1] = (i == 1) ? -1 : tetidxfactor * icavityIdx + i - 2;
			d_cavetetnext[tetidxfactor * icavityIdx + i - 1] = tetidxfactor * icavityIdx + i;
			d_cavethreadidx[tetidxfactor * icavityIdx + i - 1] = threadId;
		}
		d_cavetethead[threadId] = tetidxfactor * icavityIdx;

		d_caveoldtetlist[icavityIdx] = searchtet;
		d_caveoldtetprev[icavityIdx] = -1;
		d_caveoldtetnext[icavityIdx] = icavityIdx + 1;
		d_caveoldtethead[threadId] = icavityIdx;

		icavityIdx++;
		spintet = d_neighborlist[4 * searchtet.id + j];
		j = (spintet.ver & 3);
		for (i = 1; i < 4; i++)
		{
			neightet = d_neighborlist[4 * spintet.id + (j + i) % 4];
			d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + i - 1] = tetidxfactor * icavityIdx + i - 2;
			d_cavetetnext[tetidxfactor * icavityIdx + i - 1] = (i == 3) ? -1 : tetidxfactor * icavityIdx + i;
			d_cavethreadidx[tetidxfactor * icavityIdx + i - 1] = threadId;
		}
		d_cavetettail[threadId] = tetidxfactor * icavityIdx + 2;

		d_caveoldtetlist[icavityIdx] = spintet;
		d_caveoldtetprev[icavityIdx] = icavityIdx -1;
		d_caveoldtetnext[icavityIdx] = -1;
		d_caveoldtettail[threadId] = icavityIdx;
	}
	else if (loc == INTETRAHEDRON || loc == OUTSIDE)
	{
		int i;
		// initial cavity index
		icavityIdx = icsindex;

		// add to tet and old tet list
		for (i = 0; i < 4; i++)
		{
			neightet = d_neighborlist[4 * searchtet.id + i];
			d_cavetetlist[tetidxfactor * icavityIdx + i] = neightet;
			d_cavetetprev[tetidxfactor * icavityIdx + i] = (i == 0) ? -1 : tetidxfactor * icavityIdx + i - 1;
			d_cavetetnext[tetidxfactor * icavityIdx + i] = (i == 3) ? -1 : tetidxfactor * icavityIdx + i + 1;
			d_cavethreadidx[tetidxfactor * icavityIdx + i] = threadId;
		}
		d_cavetethead[threadId] = tetidxfactor * icavityIdx;
		d_cavetettail[threadId] = tetidxfactor * icavityIdx + 3;

		d_caveoldtetlist[icavityIdx] = searchtet;
		d_caveoldtetprev[icavityIdx] = -1;
		d_caveoldtetnext[icavityIdx] = -1;
		d_caveoldtethead[threadId] = icavityIdx;
		d_caveoldtettail[threadId] = icavityIdx;
	}

	// subcavities
	if (d_initialsubcavitysize[threadId] != 0) // when splitseg is dangling segment, this equals to 0
	{
		int iscsindex = d_initialsubcavityindices[threadId];
		int scount = 0;
		int sprev = -1;
		int iscavityIdx;

		trihandle splitsh;
		if (loc == ONEDGE)
		{
			if (threadmarker == 0)
			{
				
				int segId = d_insertidxlist[threadId];
				trihandle splitseg(segId, 0);
				cudamesh_spivot(splitseg, splitsh, d_seg2trilist);
			}
			else if (threadmarker == 1)
			{
				splitsh = d_searchsh[threadId];
			}

			// Collect all subfaces share at this edge.
			if (splitsh.id != -1)
			{
				int pa = cudamesh_sorg(splitsh, d_trifacelist);
				trihandle neighsh = splitsh;
				while (1) {
					// Initial subcavity index
					iscavityIdx = iscsindex + scount;

					// Adjust the origin of its edge to be 'pa'.
					if (cudamesh_sorg(neighsh, d_trifacelist) != pa) {
						cudamesh_sesymself(neighsh);
					}

					// add to cavesh and cavesegsh list
					d_caveshlist[iscavityIdx] = neighsh; // current tet
					d_caveshprev[iscavityIdx] = sprev; // previous
					d_caveshnext[iscavityIdx] = -1; // next, set to -1 first
					d_cavesegshlist[iscavityIdx] = neighsh; // current triface
					d_cavesegshprev[iscavityIdx] = sprev; // previous
					d_cavesegshnext[iscavityIdx] = -1; // next, set to -1 first
					if (sprev != -1)
					{
						d_caveshnext[sprev] = iscavityIdx; // previous next, set to me
						d_cavesegshnext[sprev] = iscavityIdx; // previous next, set to me
					}

					if (scount == 0)
					{
						d_caveshhead[threadId] = iscavityIdx;
						d_cavesegshhead[threadId] = iscavityIdx;
					}

					// next iteration
					sprev = iscavityIdx;

					// count this face
					scount++;

					// Go to the next face at the edge.
					cudamesh_spivotself(neighsh, d_tri2trilist);
					// Stop if all faces at the edge have been visited.
					if (neighsh.id == splitsh.id || neighsh.id == -1)
					{
						d_caveshtail[threadId] = iscavityIdx;
						d_cavesegshtail[threadId] = iscavityIdx;
						break;
					}
				} // while (1)
			}
		}
		else if (loc == ONFACE)
		{
			if (threadmarker == 1)
			{
				iscavityIdx = iscsindex;
				splitsh = d_searchsh[threadId];
				d_caveshlist[iscavityIdx] = splitsh;
				d_caveshprev[iscavityIdx] = -1;
				d_caveshnext[iscavityIdx] = -1;
				d_caveshhead[threadId] = iscavityIdx;
				d_caveshtail[threadId] = iscavityIdx;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	tethandle searchtet = d_searchtet[threadId];
	tethandle spintet, neightet;
	locateresult loc = d_pointlocation[threadId];

	// cavities
	if (d_initialcavitysize[threadId] != 0) // when it is reusecavity, this is 0
	{
		int icsindex = d_initialcavityindices[threadId];
		int count = 0;
		int icavityIdx;
		int tetidxfactor = 4;

		if (loc == ONEDGE)
		{
			spintet = searchtet;
			while (1) {
				// initial cavity index
				icavityIdx = icsindex + count;

				// add to tet list
				cudamesh_eorgoppo(spintet, neightet);
				neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
				neightet.ver = raw_epivot[neightet.ver];
				d_cavetetlist[tetidxfactor * icavityIdx] = neightet;
				d_cavetetidx[tetidxfactor * icavityIdx] = threadId;

				cudamesh_edestoppo(spintet, neightet);
				neightet = d_neighborlist[4 * neightet.id + (neightet.ver & 3)];
				neightet.ver = raw_epivot[neightet.ver];
				d_cavetetlist[tetidxfactor * icavityIdx + 1] = neightet;
				d_cavetetidx[tetidxfactor * icavityIdx + 1] = threadId;

				// add to old tet list
				d_caveoldtetlist[icavityIdx] = spintet; // current tet
				d_caveoldtetidx[icavityIdx] = threadId;

				// next iteration
				cudamesh_fnextself(spintet, d_neighborlist);
				if (spintet.id == searchtet.id)
				{
					break;
				}
				count++;
			} // while (1)
		}
		else if (loc == ONFACE)
		{
			int i, j;
			// initial cavity index
			icavityIdx = icsindex;

			// add to tet and old tet list
			j = (searchtet.ver & 3);
			for (i = 1; i < 4; i++)
			{
				neightet = d_neighborlist[4 * searchtet.id + (j + i) % 4];
				d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
				d_cavetetidx[tetidxfactor * icavityIdx + i - 1] = threadId;
			}

			d_caveoldtetlist[icavityIdx] = searchtet;
			d_caveoldtetidx[icavityIdx] = threadId;

			icavityIdx++;
			spintet = d_neighborlist[4 * searchtet.id + j];
			j = (spintet.ver & 3);
			for (i = 1; i < 4; i++)
			{
				neightet = d_neighborlist[4 * spintet.id + (j + i) % 4];
				d_cavetetlist[tetidxfactor * icavityIdx + i - 1] = neightet;
				d_cavetetidx[tetidxfactor * icavityIdx + i - 1] = threadId;
			}

			d_caveoldtetlist[icavityIdx] = spintet;
			d_caveoldtetidx[icavityIdx] = threadId;
		}
		else if (loc == INTETRAHEDRON || loc == OUTSIDE)
		{
			int i;
			// initial cavity index
			icavityIdx = icsindex;

			// add to tet and old tet list
			for (i = 0; i < 4; i++)
			{
				neightet = d_neighborlist[4 * searchtet.id + i];
				d_cavetetlist[tetidxfactor * icavityIdx + i] = neightet;
				d_cavetetidx[tetidxfactor * icavityIdx + i] = threadId;
			}

			d_caveoldtetlist[icavityIdx] = searchtet;
			d_caveoldtetidx[icavityIdx] = threadId;
		}
	}

	// subcavities
	if (d_initialsubcavitysize[threadId] != 0) // when splitseg is dangling segment, this equals to 0
	{
		int iscsindex = d_initialsubcavityindices[threadId];
		int scount = 0;
		int iscavityIdx;

		trihandle splitsh;
		if (loc == ONEDGE)
		{
			if (threadmarker == 0)
			{

				int segId = d_insertidxlist[threadId];
				trihandle splitseg(segId, 0);
				cudamesh_spivot(splitseg, splitsh, d_seg2trilist);
			}
			else if (threadmarker == 1)
			{
				splitsh = d_searchsh[threadId];
			}

			// Collect all subfaces share at this edge.
			if (splitsh.id != -1)
			{
				int pa = cudamesh_sorg(splitsh, d_trifacelist);
				trihandle neighsh = splitsh;
				while (1) {
					// Initial subcavity index
					iscavityIdx = iscsindex + scount;

					// Adjust the origin of its edge to be 'pa'.
					if (cudamesh_sorg(neighsh, d_trifacelist) != pa) {
						cudamesh_sesymself(neighsh);
					}

					// add to cavesh and cavesegsh list
					d_caveshlist[iscavityIdx] = neighsh; // current subface
					d_caveshidx[iscavityIdx] = threadId;
					d_cavesegshlist[iscavityIdx] = neighsh; // current triface
					d_cavesegshidx[iscavityIdx] = threadId;

					// count this face
					scount++;

					// Go to the next face at the edge.
					cudamesh_spivotself(neighsh, d_tri2trilist);
					// Stop if all faces at the edge have been visited.
					if (neighsh.id == splitsh.id || neighsh.id == -1)
					{
						break;
					}
				} // while (1)
			}
		}
		else if (loc == ONFACE)
		{
			if (threadmarker == 1)
			{
				iscavityIdx = iscsindex;
				splitsh = d_searchsh[threadId];
				d_caveshlist[iscavityIdx] = splitsh;
				d_caveshidx[iscavityIdx] = threadId;
			}
		}
	}
}

__global__ void kernelSetReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int oldcaveoldtetsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos + oldcaveoldtetsize];
	threadId = -threadId - 2;
	d_caveoldtetidx[pos + oldcaveoldtetsize] = threadId;
}

__global__ void kernelCheckCavetetFromReuseOldtet(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	tethandle* d_neighborlist,
	int* d_cavetetexpandsize,
	uint64* d_tetmarker,
	int oldcaveoldtetsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos + oldcaveoldtetsize];

	int cavetetexpandsize = 0;
	uint64 marker;
	int ownerId;
	bool todo;
	tethandle cavetet, neightet;

	cavetet = d_caveoldtetlist[pos + oldcaveoldtetsize];
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		if (ownerId != threadId) // boundary face
		{
			cavetetexpandsize++;
		}
	}
	d_cavetetexpandsize[pos] = cavetetexpandsize;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos + oldcaveoldtetsize];

	int startindex = d_cavetetexpandindices[pos];
	uint64 marker;
	int ownerId;
	tethandle cavetet, neightet;

	cavetet = d_caveoldtetlist[pos + oldcaveoldtetsize];
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		if (ownerId != threadId) // boundary face
		{
			d_cavetetlist[oldcavetetsize + startindex] = neightet;
			d_cavetetidx[oldcavetetsize + startindex] = threadId;
			startindex++;
		}
	}
}


__global__ void kernelInitLinklistCurPointer(
	int* d_threadlist,
	int* d_linklisthead,
	int* d_linklistcur,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	d_linklistcur[threadId] = d_linklisthead[threadId];
}

__global__ void kernelCavityRatioControl(
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId != -1) // owners of larger cavities
		d_threadmarker[threadId] = -1;
}

__global__ void kernelLargeCavityCheck(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_cavethreadidx,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			int eleidx = d_insertidxlist[threadId];
			if (threadmarker == 0)
				d_segstatus[eleidx].setAbortive(true);
			else if (threadmarker == 1)
				d_tristatus[eleidx].setAbortive(true);
			else if (threadmarker == 2)
				d_tetstatus[eleidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			int eleidx = d_insertidxlist[threadId];
			if (threadmarker == 0)
				d_segstatus[eleidx].setAbortive(true);
			else if (threadmarker == 1)
				d_tristatus[eleidx].setAbortive(true);
			else if (threadmarker == 2)
				d_tetstatus[eleidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
}

__global__ void kernelMarkCavityReuse(
	int* d_insertidxlist,
	int* d_cavetetidx,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			int eleidx = d_insertidxlist[threadId];
			if (threadmarker == 0 || threadmarker == -2)
			{
				d_segstatus[eleidx].setCavityReuse(true);
				d_threadmarker[threadId] = -2;
			}
			else if (threadmarker == 1 || threadmarker == -3)
			{
				d_tristatus[eleidx].setCavityReuse(true);
				d_threadmarker[threadId] = -3;
			}
			else if (threadmarker == 2 || threadmarker == -4)
			{
				d_tetstatus[eleidx].setCavityReuse(true);
				d_threadmarker[threadId] = -4;
			}
		}
	}
}

__global__ void kernelMarkOldtetlist(
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_insertidxlist,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == -1)
		return;

	int eleidx = d_insertidxlist[threadId];
	tethandle checktet;
	if (threadmarker == -2)
	{
		checktet = d_caveoldtetlist[pos];
		checktet.id = -(checktet.id + 1);
		checktet.ver = 0; // indicate it is a subseg
		d_caveoldtetlist[pos] = checktet;
	}
	if (threadmarker == -3) // a subface whose cavity to reuse
	{
		checktet = d_caveoldtetlist[pos];
		checktet.id = -(checktet.id + 1);
		checktet.ver = 1; // indicate it is a subface
		d_caveoldtetlist[pos] = checktet;
	}
	else if (threadmarker == -4) // a tetrahedron whose cavity to reuse
	{
		checktet = d_caveoldtetlist[pos];
		checktet.id = -(checktet.id + 1);
		checktet.ver = 2; // indicate it is a tet
		d_caveoldtetlist[pos] = checktet;
	}
}

__global__ void kernelSetRecordOldtet(
	tethandle* d_recordoldtetlist,
	int* d_recordoldtetidx,
	int* d_insertidxlist,
	int oldrecordsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	tethandle checktet = d_recordoldtetlist[pos + oldrecordsize];
	checktet.id = -checktet.id - 1;
	d_recordoldtetlist[pos + oldrecordsize] = checktet;

	int threadId = d_recordoldtetidx[pos + oldrecordsize];
	int eleidx = d_insertidxlist[threadId];
	d_recordoldtetidx[pos + oldrecordsize] = eleidx;
}

__global__ void kernelMarkLargeCavityAsLoser(
	int* d_cavetetidx,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1) // owners of large cavities
	{
		int threadmarker = d_threadmarker[threadId];
		if (threadmarker != -1)
		{
			d_threadmarker[threadId] = -1;
		}
	}
}

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
)
{

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetexpandsize = 0;
	int oldtetexpandsize = 0;
	int bdryexpandsize = 0;

	int threadId = d_cavethreadidx[pos];

	if (threadId != -1) // threadId is -1 in the unused slot
	{
		REAL* insertpt = d_insertptlist + 3 * threadId;
		int cur = cavetetcurstartindex + pos;
		tethandle cavetet = d_cavetetlist[cur];

		if (d_threadmarker[threadId] != -1) // avoid to expand loser
		{
			uint64 marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId);
			if (d_tetmarker[cavetet.id] != marker) // need to check
			{
				bool enqflag = false;
				double sign;
				// Get four endpoints of cavetet
				REAL *pts[4];
				int idx[4];
				for (int i = 0; i < 4; i++)
				{
					idx[i] = d_tetlist[4 * cavetet.id + i];
					if (idx[i] != -1)
						pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
					else
						pts[i] = NULL;
				}
				// Test if cavetet is included in the (enlarged) cavity
				if (idx[3] != -1)
				{
					sign = cudamesh_insphere_s(pts[0], pts[1], pts[2], pts[3], insertpt,
						idx[0], idx[1], idx[2], idx[3], MAXINT);
					enqflag = (sign < 0.0);
				}
				else // A hull face (must be a subface). Test its neighbor.
				{
					// We FIRST finclude it in the initial cavity if its adjacent tet is
					// not Delaunay wrt p. Will validate it later on.
					tethandle neineitet = d_neighborlist[4 * cavetet.id + 3];
					if (d_tetmarker[neineitet.id] != marker) // need to check
					{
						// Get four endpoints of neineitet
						for (int i = 0; i < 4; i++)
						{
							idx[i] = d_tetlist[4 * neineitet.id + i];
							if (idx[i] != -1)
								pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
							else
								pts[i] = NULL;
						}
						assert(idx[3] != -1);
						sign = cudamesh_insphere_s(pts[0], pts[1], pts[2], pts[3], insertpt,
							idx[0], idx[1], idx[2], idx[3], MAXINT);
						enqflag = (sign < 0.0);
					}
					else
					{
						enqflag = true;
					}
				}
				// Count size
				if (enqflag)
				{
					uint64 oldmarker = atomicMin(d_tetmarker + cavetet.id, marker);
					if (marker < oldmarker) // I winned
					{
						tetexpandsize = 3;
						oldtetexpandsize = 1;
						int old = cudamesh_getUInt64PriorityIndex(oldmarker);
						if (old != MAXUINT)
						{
							d_threadmarker[old] = -1;
						}
					}
					else if (marker > oldmarker) // I lost
					{
						d_threadmarker[threadId] = -1;
					}
				}
				else
				{
					bdryexpandsize = 1;
				}
			}
		}
	}

	d_cavetetexpandsize[pos] = tetexpandsize;
	d_caveoldtetexpandsize[pos] = oldtetexpandsize;
	d_cavebdryexpandsize[pos] = bdryexpandsize;
}

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
)
{

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetexpandsize = 0;
	int oldtetexpandsize = 0;
	int bdryexpandsize = 0;

	int cur = cavetetcurstartindex + pos;
	int threadId = d_cavetetidx[cur];

	if (threadId != -1) // threadId is -1 in the unused slot
	{
		REAL* insertpt = d_insertptlist + 3 * threadId;
		tethandle cavetet = d_cavetetlist[cur];

		if (d_threadmarker[threadId] != -1) // avoid to expand loser
		{
			uint64 marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);
			if (d_tetmarker[cavetet.id] != marker) // need to check
			{
				bool enqflag = false;
				double sign;
				// Get four endpoints of cavetet
				REAL *pts[4];
				int idx[4];
				for (int i = 0; i < 4; i++)
				{
					idx[i] = d_tetlist[4 * cavetet.id + i];
					if (idx[i] != -1)
						pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
					else
						pts[i] = NULL;
				}
				// Test if cavetet is included in the (enlarged) cavity
				if (idx[3] != -1)
				{
					sign = cudamesh_insphere_s(pts[0], pts[1], pts[2], pts[3], insertpt,
						idx[0], idx[1], idx[2], idx[3], MAXINT);
					enqflag = (sign < 0.0);
				}
				else // A hull face (must be a subface). Test its neighbor.
				{
					// We FIRST finclude it in the initial cavity if its adjacent tet is
					// not Delaunay wrt p. Will validate it later on.
					tethandle neineitet = d_neighborlist[4 * cavetet.id + 3];
					if (d_tetmarker[neineitet.id] != marker) // need to check
					{
						// Get four endpoints of neineitet
						for (int i = 0; i < 4; i++)
						{
							idx[i] = d_tetlist[4 * neineitet.id + i];
							if (idx[i] != -1)
								pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
							else
								pts[i] = NULL;
						}
						if (idx[3] == -1)
						{
							enqflag = false;
						}
						else
						{
							sign = cudamesh_insphere_s(pts[0], pts[1], pts[2], pts[3], insertpt,
								idx[0], idx[1], idx[2], idx[3], MAXINT);
							enqflag = (sign < 0.0);
						}
					}
					else
					{
						enqflag = true;
					}
				}
				// Count size
				if (enqflag)
				{
					uint64 oldmarker = atomicMax(d_tetmarker + cavetet.id, marker);
					if (marker > oldmarker) // I winned
					{
						tetexpandsize = 3;
						oldtetexpandsize = 1;
						int old = cudamesh_getUInt64PriorityIndex(oldmarker);
						if (old != 0)
						{
							d_threadmarker[old - 1] = -1;
						}
					}
					else if (marker < oldmarker) // I lost
					{
						d_threadmarker[threadId] = -1;
					}
				}
				else
				{
					bdryexpandsize = 1;
				}
			}
		}
	}

	d_cavetetexpandsize[pos] = tetexpandsize;
	d_caveoldtetexpandsize[pos] = oldtetexpandsize;
	d_cavebdryexpandsize[pos] = bdryexpandsize;
}

__global__ void  kernelCorrectExpandingSize(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId != -1 && d_threadmarker[threadId] == -1)
	{
		d_cavetetexpandsize[pos] = 0;
		d_caveoldtetexpandsize[pos] = 0;
		d_cavebdryexpandsize[pos] = 0;
	}
}

__global__ void  kernelCorrectExpandingSize(
	int* d_cavetetidx,
	int* d_cavetetexpandsize,
	int* d_caveoldtetexpandsize,
	int* d_cavebdryexpandsize,
	int* d_threadmarker,
	int cavetetcurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetidx[pos + cavetetcurstartindex];
	if (threadId != -1 && d_threadmarker[threadId] == -1)
	{
		d_cavetetexpandsize[pos] = 0;
		d_caveoldtetexpandsize[pos] = 0;
		d_cavebdryexpandsize[pos] = 0;
	}
};

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId == -1)
		return;

	int eindex;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		for (int j = 0; j < 3; j++) {
			d_cavetetthreadidx[eindex + j] = threadId;
		}
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		d_caveoldtetthreadidx[eindex] = threadId;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		d_cavebdrythreadidx[eindex] = threadId;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId == -1)
		return;
	if (d_threadmarker[threadId] == -1)
		return;

	int cur = cavetetcurstartindex + pos;
	tethandle cavetet = d_cavetetlist[cur];

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex;

		// Append cavetetlist and mark current tet
		int k = (cavetet.ver & 3); // The current face number
		tethandle neightet;
		int newid;

		if (eindex == 0 || d_cavetetthreadidx[eindex - 1] != threadId)
		{
			prev = d_cavetettail[threadId];
			d_cavetetnext[prev] = sindex; // prev must not be -1
		}
		else
			prev = sindex - 1;

		for (int j = 1; j < 4; j++) {
			neightet = d_neighborlist[4 * cavetet.id + (j + k) % 4];
			newid = sindex + j - 1;
			d_cavetetlist[newid] = neightet;
			d_cavetetprev[newid] = prev;
			d_cavetetnext[newid] = newid + 1; // set to next one first
			prev = newid;
		}

		if (eindex + 2 == cavetetexpandsize - 1 || d_cavetetthreadidx[eindex + 3] != threadId)
			d_cavetetnext[newid] = -1;
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		sindex = caveoldtetstartindex + eindex;

		if (eindex == 0 || d_caveoldtetthreadidx[eindex - 1] != threadId)
		{
			prev = d_caveoldtettail[threadId];
			d_caveoldtetnext[prev] = sindex; // prev must not be -1
		}
		else
			prev = sindex - 1;

		d_caveoldtetlist[sindex] = cavetet;
		d_caveoldtetprev[sindex] = prev;
		d_caveoldtetnext[sindex] = sindex + 1;
		
		if (eindex == caveoldtetexpandsize - 1 || d_caveoldtetthreadidx[eindex + 1] != threadId)
			d_caveoldtetnext[sindex] = -1;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;

		if (eindex == 0 || d_cavebdrythreadidx[eindex - 1] != threadId)
		{
			prev = d_cavebdrytail[threadId];
			if (prev != -1)
				d_cavebdrynext[prev] = sindex; // prev must not be -1
			if (d_cavebdryhead[threadId] == -1) // initialize cavebdry list header
				d_cavebdryhead[threadId] = sindex;
		}
		else
			prev = sindex - 1;

		cavetet.ver = raw_epivot[cavetet.ver];
		d_cavebdrylist[sindex] = cavetet;
		d_cavebdryprev[sindex] = prev;
		d_cavebdrynext[sindex] = sindex + 1;

		if (eindex == cavebdryexpandsize - 1 || d_cavebdrythreadidx[eindex + 1] != threadId)
			d_cavebdrynext[sindex] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int cur = cavetetcurstartindex + pos;

	int threadId = d_cavetetidx[cur];
	if (threadId == -1)
		return;
	if (d_threadmarker[threadId] == -1)
		return;

	tethandle cavetet = d_cavetetlist[cur];

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex;

		// Append cavetetlist and mark current tet
		int k = (cavetet.ver & 3); // The current face number
		tethandle neightet;
		int newid;

		for (int j = 1; j < 4; j++) {
			neightet = d_neighborlist[4 * cavetet.id + (j + k) % 4];
			newid = sindex + j - 1;
			d_cavetetlist[newid] = neightet;
			d_cavetetidx[newid] = threadId;
		}
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		sindex = caveoldtetstartindex + eindex;

		d_caveoldtetlist[sindex] = cavetet;
		d_caveoldtetidx[sindex] = threadId;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;

		cavetet.ver = raw_epivot[cavetet.ver];
		d_cavebdrylist[sindex] = cavetet;
		d_cavebdryidx[sindex] = threadId;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId == -1)
		return;
	if (d_threadmarker[threadId] == -1)
		return;

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex + 2;
		if (d_cavetetnext[sindex] == -1)
			d_cavetettail[threadId] = sindex;
	}

	if (d_caveoldtetexpandsize[pos] != 0)
	{
		eindex = d_caveoldtetexpandindices[pos];
		sindex = caveoldtetstartindex + eindex;
		if (d_caveoldtetnext[sindex] == -1)
			d_caveoldtettail[threadId] = sindex;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;
		if (d_cavebdrynext[sindex] == -1)
			d_cavebdrytail[threadId] = sindex;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int i = d_caveoldtethead[threadId];
	int old;
	tethandle cavetet;
	trihandle checkseg;
	while (i != -1)
	{
		cavetet = d_caveoldtetlist[i];
		for (int j = 0; j < 6; j++)
		{
			checkseg = d_tet2seglist[6 * cavetet.id + j];
			if (checkseg.id != -1)
			{
				old = atomicMin(d_segmarker + checkseg.id, threadId);
				if (old < threadId)
					d_threadmarker[threadId] = -1;
				else if (old > threadId && old != MAXINT)
					d_threadmarker[old] = -1;
			}
		}
		i = d_caveoldtetnext[i];
	}
}

__global__ void kernelMarkCavityAdjacentSubsegs(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_priority,
	uint64* d_segmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	uint64 marker, oldmarker;
	tethandle cavetet;
	trihandle checkseg;

	marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);

	cavetet = d_caveoldtetlist[pos];
	for (int j = 0; j < 6; j++)
	{
		checkseg = d_tet2seglist[6 * cavetet.id + j];
		if (checkseg.id != -1)
		{
			oldmarker = atomicMax(d_segmarker + checkseg.id, marker);
			if (marker > oldmarker) // I winned
			{
				int old = cudamesh_getUInt64PriorityIndex(oldmarker);
				if (old != 0)
				{
					d_threadmarker[old - 1] = -1;
				}
			}
			else if (marker < oldmarker) // I lost
			{
				d_threadmarker[threadId] = -1;
			}
		}
	}
}


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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int cavetetsegsize = 0;
	if (d_threadmarker[threadId] != -1)
	{
		int i = d_caveoldtethead[threadId];
		tethandle cavetet;
		trihandle checkseg;
		while (i != -1)
		{
			cavetet = d_caveoldtetlist[i];
			for (int j = 0; j < 6; j++)
			{
				checkseg = d_tet2seglist[6 * cavetet.id + j];
				if (checkseg.id != -1)
				{
					if (d_segmarker[checkseg.id] == threadId)
					{
						cavetetsegsize++;
						d_segmarker[checkseg.id] = MAXINT; // Mark as counted
					}
				}
			}
			i = d_caveoldtetnext[i];
		}
	}
	d_cavetetsegsize[pos] = cavetetsegsize;
}

__global__ void kernelCountCavitySubsegs_Phase1(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	uint64* d_segmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	if (d_threadmarker[threadId] != -1)
	{
		tethandle cavetet;
		trihandle checkseg;
		cavetet = d_caveoldtetlist[pos];
		for (int j = 0; j < 6; j++)
		{
			checkseg = d_tet2seglist[6 * cavetet.id + j];
			if (checkseg.id != -1)
			{
				atomicMax(d_segmarker + checkseg.id, pos + 1);
			}
		}
	}
}

__global__ void kernelCountCavitySubsegs_Phase2(
	trihandle* d_tet2seglist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_cavetetsegsize,
	uint64* d_segmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	int cavetetsegsize = 0;
	if (d_threadmarker[threadId] != -1)
	{
		tethandle cavetet;
		trihandle checkseg;
		cavetet = d_caveoldtetlist[pos];
		for (int j = 0; j < 6; j++)
		{
			checkseg = d_tet2seglist[6 * cavetet.id + j];
			if (checkseg.id != -1)
			{
				if(d_segmarker[checkseg.id] == pos + 1) // I should count this subseg
					cavetetsegsize++;
			}
		}
	}
	d_cavetetsegsize[pos] = cavetetsegsize;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	if (d_cavetetsegsize[pos] == 0)
		return;

	int sindex = d_cavetetsegindices[pos];
	d_cavetetseghead[threadId] = sindex;

	int i = d_caveoldtethead[threadId];
	tethandle cavetet;
	trihandle checkseg;
	int index, count = 0, prev = -1;
	while (i != -1)
	{
		cavetet = d_caveoldtetlist[i];
		for (int j = 0; j < 6; j++)
		{
			checkseg = d_tet2seglist[6 * cavetet.id + j];
			if (checkseg.id != -1)
			{
				if (d_segmarker[checkseg.id] == MAXINT)
				{
					d_segmarker[checkseg.id] = -2; // Mark as appended

					index = sindex + count;
					d_cavetetseglist[index] = checkseg;
					d_cavetetsegprev[index] = prev;
					d_cavetetsegnext[index] = -1;
					if (prev != -1)
						d_cavetetsegnext[prev] = index;
					count++;
					prev = index;
				}
			}
		}
		i = d_caveoldtetnext[i];
		if (i == -1) // reached the end
		{
			d_cavetetsegtail[threadId] = index;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_cavetetsegsize[pos] == 0)
		return;

	int threadId = d_caveoldtetidx[pos];
	int sindex = d_cavetetsegindices[pos];

	tethandle cavetet;
	trihandle checkseg;
	int index, count = 0;

	cavetet = d_caveoldtetlist[pos];
	for (int j = 0; j < 6; j++)
	{
		checkseg = d_tet2seglist[6 * cavetet.id + j];
		if (checkseg.id != -1)
		{
			if (d_segmarker[checkseg.id] == pos + 1)
			{
				index = sindex + count;
				d_cavetetseglist[index] = checkseg;
				d_cavetetsegidx[index] = threadId;
				count++;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker <= 0) // loser or subsegment
		return;

	REAL *insertpt = d_insertptlist + 3 * threadId;
	int ipa, ipb, encpt;
	REAL *pa, *pb;
	trihandle paryseg;
	
	bool flag = false;
	int i = d_cavetetseghead[threadId];
	while (i != -1)
	{
		paryseg = d_cavetetseglist[i];
		ipa = d_seglist[3 * paryseg.id + 0];
		ipb = d_seglist[3 * paryseg.id + 1];
		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		if (checkseg4encroach(pa, pb, insertpt)) // encroached
		{
			flag = true;
			if (!d_segstatus[paryseg.id].isAbortive())
			{
				d_segencmarker[paryseg.id] = MAXINT;
				d_threadmarker[threadId] = -1;
				break;
			}
		}
		i = d_cavetetsegnext[i];
	}

	if (flag && d_threadmarker[threadId] != -1) // segments encroached are all abortive
	{
		int eleidx = d_insertidxlist[threadId];
		if (threadmarker == 1)
		{
			d_tristatus[eleidx].setAbortive(true);
		}
		else if (threadmarker == 2)
		{
			d_tetstatus[eleidx].setAbortive(true);
		}
		d_threadmarker[threadId] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetsegidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker <= 0) // loser or subsegment
		return;

	REAL *insertpt = d_insertptlist + 3 * threadId;
	int ipa, ipb, encpt;
	REAL *pa, *pb;
	trihandle paryseg;

	paryseg = d_cavetetseglist[pos];
	ipa = d_seglist[3 * paryseg.id + 0];
	ipb = d_seglist[3 * paryseg.id + 1];
	pa = cudamesh_id2pointlist(ipa, d_pointlist);
	pb = cudamesh_id2pointlist(ipb, d_pointlist);
	if (checkseg4encroach(pa, pb, insertpt)) // encroached
	{
		d_encroachmentmarker[pos] = 1;
		if (!d_segstatus[paryseg.id].isAbortive())
		{
			int oldmarker = atomicMin(d_threadmarker + threadId, -1);
			if (oldmarker >= 0)
				d_segencmarker[paryseg.id] = MAXINT;
		}
	}
}

__global__ void kernelCheckSegmentEncroachment_Phase2(
	int* d_insertidxlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_cavetetsegidx,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetsegidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker <= 0) // loser or subsegment
		return;

	int encmarker = d_encroachmentmarker[pos];
	if(encmarker)
	{
		int eleidx = d_insertidxlist[threadId];
		if (threadmarker == 1)
		{
			d_tristatus[eleidx].setAbortive(true);
		}
		else if (threadmarker == 2)
		{
			d_tetstatus[eleidx].setAbortive(true);
		}
		d_threadmarker[threadId] = -1;
	}
}

__global__ void kernelMarkCavityAdjacentSubfaces(
	int* d_threadlist,
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetnext,
	int* d_caveoldtethead,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int i = d_caveoldtethead[threadId];
	int old;
	tethandle cavetet;
	trihandle checksh;
	while (i != -1)
	{
		cavetet = d_caveoldtetlist[i];
		for (int j = 0; j < 4; j++)
		{
			checksh = d_tet2trilist[4 * cavetet.id + j];
			if (checksh.id != -1)
			{
				old = atomicMin(d_trimarker + checksh.id, threadId);
				if (old < threadId)
					d_threadmarker[threadId] = -1;
				else if (old > threadId && old != MAXINT)
					d_threadmarker[old] = -1;
			}
		}
		i = d_caveoldtetnext[i];
	}
}

__global__ void kernelMarkCavityAdjacentFaces(
	trihandle* d_tet2trilist,
	tethandle* d_neighborlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_priority,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	uint64 marker;
	int ownerId;
	tethandle cavetet, neightet;
	trihandle checksh;

	cavetet = d_caveoldtetlist[pos];
	for (int j = 0; j < 4; j++)
	{
		// check neighbor
		cavetet.ver = j;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		marker = d_tetmarker[neightet.id];
		ownerId = cudamesh_getUInt64PriorityIndex(marker) - 1;

		if (ownerId != threadId) // boundary face
		{
			if (ownerId != -1 && d_threadmarker[ownerId] >= 0
				&& (d_priority[threadId] < d_priority[ownerId] ||
				(d_priority[threadId] == d_priority[ownerId] && threadId < ownerId))) // I lost
			{
				d_threadmarker[threadId] = -1;
				return;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int cavetetshsize = 0;

	if (d_threadmarker[threadId] != -1)
	{
		int i = d_caveoldtethead[threadId];
		tethandle cavetet;
		trihandle checksh;
		while (i != -1)
		{
			cavetet = d_caveoldtetlist[i];
			for (int j = 0; j < 4; j++)
			{
				checksh = d_tet2trilist[4 * cavetet.id + j];
				if (checksh.id != -1)
				{
					if (d_trimarker[checksh.id] == threadId)
					{
						cavetetshsize++;
						d_trimarker[checksh.id] = MAXINT;
					}
				}
			}
			i = d_caveoldtetnext[i];
		}
	}
	d_cavetetshsize[pos] = cavetetshsize;
}

__global__ void kernelCountCavitySubfaces_Phase1(
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	if (d_threadmarker[threadId] != -1)
	{
		tethandle cavetet;
		trihandle checksh;
		cavetet = d_caveoldtetlist[pos];
		for (int j = 0; j < 4; j++)
		{
			checksh = d_tet2trilist[4 * cavetet.id + j];
			if (checksh.id != -1)
			{
				atomicMax(d_trimarker + checksh.id, pos + 1);
			}
		}
	}
}

__global__ void kernelCountCavitySubfaces_Phase2(
	trihandle* d_tet2trilist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int* d_cavetetshsize,
	int* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	int cavetetshsize = 0;

	if (d_threadmarker[threadId] != -1)
	{
		tethandle cavetet;
		trihandle checksh;
		cavetet = d_caveoldtetlist[pos];
		for (int j = 0; j < 4; j++)
		{
			checksh = d_tet2trilist[4 * cavetet.id + j];
			if (checksh.id != -1)
			{
				if (d_trimarker[checksh.id] == pos + 1)
					cavetetshsize++;
			}
		}
	}

	d_cavetetshsize[pos] = cavetetshsize;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	if (d_cavetetshsize[pos] == 0)
		return;

	int sindex = d_cavetetshindices[pos];
	d_cavetetshhead[threadId] = sindex;

	int i = d_caveoldtethead[threadId];
	tethandle cavetet;
	trihandle checksh;
	int index, count = 0, prev = -1;
	while (i != -1)
	{
		cavetet = d_caveoldtetlist[i];
		for (int j = 0; j < 4; j++)
		{
			checksh = d_tet2trilist[4 * cavetet.id + j];
			if (checksh.id != -1)
			{
				if (d_trimarker[checksh.id] == MAXINT)
				{
					d_trimarker[checksh.id] = -2; // Mark as appended

					index = sindex + count;
					d_cavetetshlist[index] = checksh;
					d_cavetetshprev[index] = prev;
					d_cavetetshnext[index] = -1;
					if (prev != -1)
						d_cavetetshnext[prev] = index;
					count++;
					prev = index;
				}
			}
		}
		i = d_caveoldtetnext[i];
		if (i == -1) // reached the end
		{
			d_cavetetshtail[threadId] = index;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_cavetetshsize[pos] == 0)
		return;

	int threadId = d_caveoldtetidx[pos];
	int sindex = d_cavetetshindices[pos];

	tethandle cavetet;
	trihandle checksh;
	int index, count = 0;

	cavetet = d_caveoldtetlist[pos];
	for (int j = 0; j < 4; j++)
	{
		checksh = d_tet2trilist[4 * cavetet.id + j];
		if (checksh.id != -1)
		{
			if (d_trimarker[checksh.id] == pos + 1)
			{
				index = sindex + count;
				d_cavetetshlist[index] = checksh;
				d_cavetetshidx[index] = threadId;
				count++;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 2) // not a tetrahedron
		return;

	locateresult loc = d_pointlocation[threadId];
	REAL *insertpt = d_insertptlist + 3 * threadId;
	REAL *pa, *pb, *pc;
	trihandle parysh;

	bool flag = false;
	int i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		parysh = d_cavetetshlist[i];
		pa = cudamesh_id2pointlist(d_trifacelist[3 * parysh.id + 0], d_pointlist);
		pb = cudamesh_id2pointlist(d_trifacelist[3 * parysh.id + 1], d_pointlist);
		pc = cudamesh_id2pointlist(d_trifacelist[3 * parysh.id + 2], d_pointlist);
		if (checkface4encroach(pa, pb, pc, insertpt)) // encroached
		{
			flag = true;
			if (!d_tristatus[parysh.id].isAbortive())
			{
				d_subfaceencmarker[parysh.id] = MAXINT;
				d_threadmarker[threadId] = -1;
				break;
			}
		}
		i = d_cavetetshnext[i];
	}

	if (loc == OUTSIDE || (flag && d_threadmarker[threadId] != -1)) 
	{
		// subfaces encroached are all abortive or points are outside the domain
		int insertidx = d_insertidxlist[threadId];
		d_tetstatus[insertidx].setAbortive(true);
		d_threadmarker[threadId] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 2) // not a tetrahedron
		return;

	REAL *insertpt = d_insertptlist + 3 * threadId;
	REAL *pa, *pb, *pc;
	trihandle parysh;

	parysh = d_cavetetshlist[pos];
	pa = cudamesh_id2pointlist(d_trifacelist[3 * parysh.id + 0], d_pointlist);
	pb = cudamesh_id2pointlist(d_trifacelist[3 * parysh.id + 1], d_pointlist);
	pc = cudamesh_id2pointlist(d_trifacelist[3 * parysh.id + 2], d_pointlist);
	if (checkface4encroach(pa, pb, pc, insertpt)) // encroached
	{
		d_encroachmentmarker[pos] = 1;
		if (!d_tristatus[parysh.id].isAbortive())
		{
			int oldmarker = atomicMin(d_threadmarker + threadId, -1);
			if(oldmarker >= 0)
				d_subfaceencmarker[parysh.id] = MAXINT;
		}
	}
}

__global__ void kernelCheckSubfaceEncroachment_Phase2(
	int* d_insertidxlist,
	locateresult* d_pointlocation,
	tetstatus* d_tetstatus,
	int* d_cavetetshidx,
	int* d_encroachmentmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 2) // not a tetrahedron
		return;

	locateresult loc = d_pointlocation[threadId];
	int encmarker = d_encroachmentmarker[pos];

	if (loc == OUTSIDE || encmarker)
	{
		// subfaces encroached are all abortive or points are outside the domain
		int insertidx = d_insertidxlist[threadId];
		d_tetstatus[insertidx].setAbortive(true);
		d_threadmarker[threadId] = -1;
	}
}


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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	REAL* insertpt = d_insertptlist + 3 * threadId;
	int cur = d_caveshcur[threadId];
	if (cur == -1) // this means that caveshlist is empty
		return;

	trihandle checksh = d_caveshlist[cur];
	trihandle neighsh;
	tethandle neightet;
	REAL sign;
	REAL* pt[3];
	int flag[3] = {0, 0, 0};

	int shexpandsize = 0;
	//assert(d_trimarker[checksh.id] == threadId);
	uint64 marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId);
	for (int j = 0; j < 3; j++)
	{
		if (!cudamesh_isshsubseg(checksh, d_tri2seglist))
		{
			cudamesh_spivot(checksh, neighsh, d_tri2trilist);
			//assert(neighsh.id != -1);
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId)
			{
				cudamesh_stpivot(neighsh, neightet, d_tri2tetlist);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
				{
					cudamesh_fsymself(neightet, d_neighborlist);
					if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
					{
						pt[0] = cudamesh_id2pointlist(cudamesh_sorg(neighsh, d_trifacelist), d_pointlist);
						pt[1] = cudamesh_id2pointlist(cudamesh_sdest(neighsh, d_trifacelist), d_pointlist);
						pt[2] = cudamesh_id2pointlist(cudamesh_sapex(neighsh, d_trifacelist), d_pointlist);
						sign = cudamesh_incircle3d(pt[0], pt[1], pt[2], insertpt);
						if (sign < 0)
						{
							atomicMin(d_trimarker + neighsh.id, marker);
							shexpandsize++;
							flag[j] = 1;
						}
					}
				}
			}
		}
		cudamesh_senextself(checksh);
	}
	d_caveshexpandsize[pos] = shexpandsize;
	if (shexpandsize > 0)
	{
		for (int j = 0; j < 3; j++)
		{
			if ((flag[j] == 1 && shexpandsize == 1) || (flag[j] == 0 && shexpandsize == 2))
			{
				d_caveshexpandflag[pos] = j;
				break;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int cur = caveshcurstartindex + pos;
	int threadId = d_caveshidx[cur];
	if (d_threadmarker[threadId] == -1)
		return;

	REAL* insertpt = d_insertptlist + 3 * threadId;
	trihandle checksh = d_caveshlist[cur];
	trihandle neighsh;
	tethandle neightet;
	REAL sign;
	REAL* pt[3];
	int flag[3] = { 0, 0, 0 };

	int shexpandsize = 0;
	uint64 marker = cudamesh_encodeUInt64Priority(d_priority[threadId], threadId + 1);
	for (int j = 0; j < 3; j++)
	{
		if (!cudamesh_isshsubseg(checksh, d_tri2seglist))
		{
			cudamesh_spivot(checksh, neighsh, d_tri2trilist);
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId + 1)
			{
				cudamesh_stpivot(neighsh, neightet, d_tri2tetlist);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
				{
					cudamesh_fsymself(neightet, d_neighborlist);
					if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
					{
						pt[0] = cudamesh_id2pointlist(cudamesh_sorg(neighsh, d_trifacelist), d_pointlist);
						pt[1] = cudamesh_id2pointlist(cudamesh_sdest(neighsh, d_trifacelist), d_pointlist);
						pt[2] = cudamesh_id2pointlist(cudamesh_sapex(neighsh, d_trifacelist), d_pointlist);
						sign = cudamesh_incircle3d(pt[0], pt[1], pt[2], insertpt);
						if (sign < 0)
						{
							atomicMax(d_trimarker + neighsh.id, marker);
							shexpandsize++;
							flag[j] = 1;
						}
					}
				}
			}
		}
		cudamesh_senextself(checksh);
	}
	d_caveshexpandsize[pos] = shexpandsize;
	if (shexpandsize > 0)
	{
		for (int j = 0; j < 3; j++)
		{
			if ((flag[j] == 1 && shexpandsize == 1) || (flag[j] == 0 && shexpandsize == 2))
			{
				d_caveshexpandflag[pos] = j;
				break;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int cur = d_caveshcur[threadId];
	if (cur == -1)
	{
		d_threadfinishmarker[threadId] = -1;
		return;
	}

	trihandle checksh = d_caveshlist[cur];
	trihandle neighsh;

	int sindex;
	int caveshexpandsize = d_caveshexpandsize[pos];
	int caveshexpandflag = d_caveshexpandflag[pos];
	if (caveshexpandsize != 0)
	{
		sindex = caveshstartindex + d_caveshexpandindices[pos];
		int prev = d_caveshtail[threadId];

		int newid = sindex;
		for (int j = 0; j < 3; j++)
		{
			if ((caveshexpandsize == 1 && j == caveshexpandflag) ||
				(caveshexpandsize == 2 && j != caveshexpandflag) ||
				caveshexpandsize == 3)
			{
				cudamesh_spivot(checksh, neighsh, d_tri2trilist);
				d_caveshlist[newid] = neighsh;
				d_caveshprev[newid] = prev;
				d_caveshnext[newid] = -1;
				if (prev != -1)
					d_caveshnext[prev] = newid;
				prev = newid++;
			}
			cudamesh_senextself(checksh);
		}
		d_caveshtail[threadId] = newid - 1;

		// Update current linklist pointer to next one
		d_caveshcur[threadId] = d_caveshnext[cur];
	}
	else
	{
		if (cur == d_caveshtail[threadId])
			d_threadfinishmarker[threadId] = -1;
		else
			d_caveshcur[threadId] = d_caveshnext[cur];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int caveshexpandsize = d_caveshexpandsize[pos];
	if (caveshexpandsize == 0)
		return;

	int caveshexpandflag = d_caveshexpandflag[pos];
	int cur = caveshcurstartindex + pos;
	int threadId = d_caveshidx[cur];

	trihandle checksh = d_caveshlist[cur];
	trihandle neighsh;

	int newid = caveshstartindex + d_caveshexpandindices[pos];
	for (int j = 0; j < 3; j++)
	{
		if ((caveshexpandsize == 1 && j == caveshexpandflag) ||
			(caveshexpandsize == 2 && j != caveshexpandflag) ||
			caveshexpandsize == 3)
		{
			cudamesh_spivot(checksh, neighsh, d_tri2trilist);
			d_caveshlist[newid] = neighsh;
			d_caveshidx[newid] = threadId;
			newid++;
		}
		cudamesh_senextself(checksh);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	REAL* insertpt = d_insertptlist + 3 * threadId;

	trihandle parysh;
	tethandle neightet;
	int cavebdryexpandsize = 0;
	int cutcount = 0;
	double ori;
	int i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		parysh = d_cavetetshlist[i];
		cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
		{
			cudamesh_fsymself(neightet, d_neighborlist);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
			{
				// Found a subface inside cavity
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) != threadId)
				{

					if (cudamesh_oppo(neightet, d_tetlist) != -1)
					{
						cudamesh_fsymself(neightet, d_neighborlist);
					}
					if (cudamesh_oppo(neightet, d_tetlist) != -1)
					{
						int idx[3];
						REAL* pt[3];
						idx[0] = cudamesh_org(neightet, d_tetlist);
						idx[1] = cudamesh_dest(neightet, d_tetlist);
						idx[2] = cudamesh_apex(neightet, d_tetlist);
						for (int j = 0; j < 3; j++)
						{
							pt[j] = cudamesh_id2pointlist(idx[j], d_pointlist);
						}
						ori = cuda_orient3d(pt[0], pt[1], pt[2], insertpt);
						
						if (ori < 0)
						{
							cudamesh_fsymself(neightet, d_neighborlist);
							ori = -ori;
						}
					}
					else
					{
						ori = 1;
					}
					// unmark and record this tet if it is either invisible by or coplanar with p
					if (ori >= 0)
					{
						d_tetmarker[neightet.id] = MAXULL; // unmark this tet
						d_cavetetshmarker[i] = 0; // mark this subface
						d_cavetetshflag[i] = neightet;
						cutcount++;
						cavebdryexpandsize += 4;
					}
				}
			}
		}
		i = d_cavetetshnext[i];
	}
	d_cavebdryexpandsize[pos] = cavebdryexpandsize;
	d_cutcount[threadId] = cutcount;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	REAL* insertpt = d_insertptlist + 3 * threadId;

	trihandle parysh;
	tethandle neightet;
	int cavebdryexpandsize = 0;
	double ori;

	parysh = d_cavetetshlist[pos];
	cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
	if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
	{
		cudamesh_fsymself(neightet, d_neighborlist);
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
		{
			// Found a subface inside cavity
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) != threadId + 1)
			{

				if (cudamesh_oppo(neightet, d_tetlist) != -1)
				{
					cudamesh_fsymself(neightet, d_neighborlist);
				}
				if (cudamesh_oppo(neightet, d_tetlist) != -1)
				{
					int idx[3];
					REAL* pt[3];
					idx[0] = cudamesh_org(neightet, d_tetlist);
					idx[1] = cudamesh_dest(neightet, d_tetlist);
					idx[2] = cudamesh_apex(neightet, d_tetlist);
					for (int j = 0; j < 3; j++)
					{
						pt[j] = cudamesh_id2pointlist(idx[j], d_pointlist);
					}
					ori = cuda_orient3d(pt[0], pt[1], pt[2], insertpt);

					if (ori < 0)
					{
						cudamesh_fsymself(neightet, d_neighborlist);
						ori = -ori;
					}
				}
				else
				{
					ori = 1;
				}
				// unmark and record this tet if it is either invisible by or coplanar with p
				if (ori >= 0)
				{
					d_tetmarker[neightet.id] = 0; // unmark this tet
					d_cavetetshflag[pos] = neightet;
					cavebdryexpandsize = 4;
				}
			}
		}
	}
	d_cavebdryexpandsize[pos] = cavebdryexpandsize;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;
	if (d_cavebdryexpandsize[pos] == 0)
		return;

	int threadId = d_threadlist[pos];

	tethandle neightet, neineitet;
	int sindex = d_cavebdryexpandindices[pos];
	int prev = d_cavebdrytail[threadId];
	int newid = cavebdrystartindex + sindex;
	int i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		if (d_cavetetshmarker[i] == 0) // Need to append
		{
			neightet = d_cavetetshflag[i];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavebdrylist[newid] = neightet;
			d_cavebdryprev[newid] = prev;
			d_cavebdrynext[newid] = -1; // set to -1 first
			if (prev != -1)
				d_cavebdrynext[prev] = newid;
			prev = newid;
			newid++;
			for (int j = 0; j < 3; j++)
			{
				cudamesh_esym(neightet, neineitet);
				neineitet.ver = raw_epivot[neineitet.ver];
				d_cavebdrylist[newid] = neineitet;
				d_cavebdryprev[newid] = prev;
				d_cavebdrynext[newid] = -1; // set to -1 first
				if (prev != -1)
					d_cavebdrynext[prev] = newid;
				prev = newid;
				newid++;
				cudamesh_enextself(neightet);
			}
		}
		i = d_cavetetshnext[i];
		if (i == -1)
		{
			d_cavebdrytail[threadId] = newid - 1;
		}
	}
}

__global__ void kernelCavityBoundarySubfacesAppend(
	int* d_cavetetshidx,
	tethandle* d_cavetetshflag,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;
	if (d_cavebdryexpandsize[pos] == 0)
		return;

	int threadId = d_cavetetshidx[pos];

	tethandle neightet, neineitet;
	int sindex = d_cavebdryexpandindices[pos];
	int newid = cavebdrystartindex + sindex;

	neightet = d_cavetetshflag[pos];
	neightet.ver = raw_epivot[neightet.ver];
	d_cavebdrylist[newid] = neightet;
	d_cavebdryidx[newid] = threadId;
	newid++;
	for (int j = 0; j < 3; j++)
	{
		cudamesh_esym(neightet, neineitet);
		neineitet.ver = raw_epivot[neineitet.ver];
		d_cavebdrylist[newid] = neineitet;
		d_cavebdryidx[newid] = threadId;
		newid++;
		cudamesh_enextself(neightet);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	REAL* insertpt = d_insertptlist + 3 * threadId;
	REAL *pa, *pb, *pc;
	int ia, ib, ic;

	trihandle paryseg;
	tethandle neightet, spintet, neineitet;
	int cavebdryexpandsize = 0;
	int cutcount = 0;
	double ori;
	int i = d_cavetetseghead[threadId];
	int j;
	while (i != -1)
	{
		paryseg = d_cavetetseglist[i];
		if (d_segmarker[paryseg.id] != threadId) // not a splitting segment
		{
			cudamesh_sstpivot1(paryseg, neightet, d_seg2tetlist);
			{
				int pa, pb, pc, pd;
				pa = cudamesh_sorg(paryseg, d_seglist);
				pb = cudamesh_sdest(paryseg, d_seglist);
				pc = cudamesh_org(neightet, d_tetlist);
				pd = cudamesh_dest(neightet, d_tetlist);
				if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
				{

				}
			}
			spintet = neightet;
			while (1)
			{
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId)
					break;
				cudamesh_fnextself(spintet, d_neighborlist);
				if (spintet.id == neightet.id)
					break;
			}
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) == threadId) // This segment is inside cavity
			{
				// Find an adjacent tet at this segment such that both faces
				//   at this segment are not visible by p.
				ia = cudamesh_org(neightet, d_tetlist);
				ib = cudamesh_dest(neightet, d_tetlist);
				pa = cudamesh_id2pointlist(ia, d_pointlist);
				pb = cudamesh_id2pointlist(ib, d_pointlist);
				spintet = neightet;
				j = 0;
				while (1)
				{
					ic = cudamesh_apex(spintet, d_tetlist);
					if (ic != -1)
					{
						pc = cudamesh_id2pointlist(ic, d_pointlist);
						ori = cuda_orient3d(pa, pb, pc, insertpt);
						if (ori >= 0)
						{
							// Not visible. Check another face in this tet.
							cudamesh_esym(spintet, neineitet);
							ic = cudamesh_apex(neineitet, d_tetlist);
							if (ic != -1)
							{
								pc = cudamesh_id2pointlist(ic, d_pointlist);
								ori = cuda_orient3d(pb, pa, pc, insertpt);
								if (ori >= 0)
								{
									// Not visible. Found this face.
									j = 1; // Flag that it is found.
									break;
								}
							}
						}
					}
					cudamesh_fnextself(spintet, d_neighborlist);
					if (spintet.id == neightet.id)
						break;
				}
				if (j == 0)
				{
					//printf("threadId #%d: Subseg check error - Couldn't find the tet to be unmarked!\n", threadId);
				}
				neightet = spintet;
				d_tetmarker[neightet.id] = MAXULL; // unmark this tet
				d_cavetetsegmarker[i] = 0; // mark this subseg
				d_cavetetsegflag[i] = neightet;
				cutcount++;
				cavebdryexpandsize += 4;
			}
		}
		i = d_cavetetsegnext[i];
	}
	d_cavebdryexpandsize[pos] = cavebdryexpandsize;
	d_cutcount[threadId] += cutcount;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetsegidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	REAL* insertpt = d_insertptlist + 3 * threadId;
	REAL *pa, *pb, *pc;
	int ia, ib, ic;

	trihandle paryseg;
	tethandle neightet, spintet, neineitet;
	int cavebdryexpandsize = 0;
	double ori;
	int j;

	paryseg = d_cavetetseglist[pos];
	if (d_segmarker[paryseg.id] != threadId + 1) // not a splitting segment
	{
		cudamesh_sstpivot1(paryseg, neightet, d_seg2tetlist);
		{
			int pa, pb, pc, pd;
			pa = cudamesh_sorg(paryseg, d_seglist);
			pb = cudamesh_sdest(paryseg, d_seglist);
			pc = cudamesh_org(neightet, d_tetlist);
			pd = cudamesh_dest(neightet, d_tetlist);
			if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
			{

			}
		}
		spintet = neightet;
		while (1)
		{
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId + 1)
				break;
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == neightet.id)
				break;
		}
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) == threadId + 1) // This segment is inside cavity
		{
			// Find an adjacent tet at this segment such that both faces
			//   at this segment are not visible by p.
			ia = cudamesh_org(neightet, d_tetlist);
			ib = cudamesh_dest(neightet, d_tetlist);
			pa = cudamesh_id2pointlist(ia, d_pointlist);
			pb = cudamesh_id2pointlist(ib, d_pointlist);
			spintet = neightet;
			j = 0;
			while (1)
			{
				ic = cudamesh_apex(spintet, d_tetlist);
				if (ic != -1)
				{
					pc = cudamesh_id2pointlist(ic, d_pointlist);
					ori = cuda_orient3d(pa, pb, pc, insertpt);
					if (ori >= 0)
					{
						// Not visible. Check another face in this tet.
						cudamesh_esym(spintet, neineitet);
						ic = cudamesh_apex(neineitet, d_tetlist);
						if (ic != -1)
						{
							pc = cudamesh_id2pointlist(ic, d_pointlist);
							ori = cuda_orient3d(pb, pa, pc, insertpt);
							if (ori >= 0)
							{
								// Not visible. Found this face.
								j = 1; // Flag that it is found.
								break;
							}
						}
					}
				}
				cudamesh_fnextself(spintet, d_neighborlist);
				if (spintet.id == neightet.id)
					break;
			}
			if (j == 0)
			{
				//printf("threadId #%d: Subseg check error - Couldn't find the tet to be unmarked!\n", threadId);
			}
			neightet = spintet;
			d_tetmarker[neightet.id] = 0; // unmark this tet
			d_cavetetsegflag[pos] = neightet;
			cavebdryexpandsize += 4;
		}
	}
	d_cavebdryexpandsize[pos] = cavebdryexpandsize;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;
	if (d_cavebdryexpandsize[pos] == 0)
		return;

	int threadId = d_threadlist[pos];

	tethandle neightet, neineitet;
	int sindex = d_cavebdryexpandindices[pos];
	int prev = d_cavebdrytail[threadId];
	int newid = cavebdrystartindex + sindex;
	int i = d_cavetetseghead[threadId];
	while (i != -1)
	{
		if (d_cavetetsegmarker[i] == 0) // Need to append
		{
			neightet = d_cavetetsegflag[i];
			neightet.ver = raw_epivot[neightet.ver];
			d_cavebdrylist[newid] = neightet;
			d_cavebdryprev[newid] = prev;
			d_cavebdrynext[newid] = -1; // set to -1 first
			if (prev != -1)
				d_cavebdrynext[prev] = newid;
			prev = newid;
			newid++;
			for (int j = 0; j < 3; j++)
			{
				cudamesh_esym(neightet, neineitet);
				neineitet.ver = raw_epivot[neineitet.ver];
				d_cavebdrylist[newid] = neineitet;
				d_cavebdryprev[newid] = prev;
				d_cavebdrynext[newid] = -1; // set to -1 first
				if (prev != -1)
					d_cavebdrynext[prev] = newid;
				prev = newid;
				newid++;
				cudamesh_enextself(neightet);
			}
		}
		i = d_cavetetsegnext[i];
		if (i == -1)
		{
			d_cavebdrytail[threadId] = newid - 1;
		}
	}
}

__global__ void kernelCavityBoundarySubsegsAppend(
	int* d_cavetetsegidx,
	tethandle* d_cavetetsegflag,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;
	if (d_cavebdryexpandsize[pos] == 0)
		return;

	int threadId = d_cavetetsegidx[pos];

	tethandle neightet, neineitet;
	int sindex = d_cavebdryexpandindices[pos];
	int newid = cavebdrystartindex + sindex;

	neightet = d_cavetetsegflag[pos];
	neightet.ver = raw_epivot[neightet.ver];
	d_cavebdrylist[newid] = neightet;
	d_cavebdryidx[newid] = threadId;
	newid++;
	for (int j = 0; j < 3; j++)
	{
		cudamesh_esym(neightet, neineitet);
		neineitet.ver = raw_epivot[neineitet.ver];
		d_cavebdrylist[newid] = neineitet;
		d_cavebdryidx[newid] = threadId;
		newid++;
		cudamesh_enextself(neightet);
	}
}

__global__ void kernelUpdateCavity2StarShapedSortOutBoundaryListCount(
	int* d_threadlist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_cavecount,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int count = 0;
	int threadId = d_threadlist[pos];
	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		count += 1;
		i = d_cavebdrynext[i];
	}
	d_cavecount[pos] = count;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int sindex = d_expandindices[pos];
	int prev = -1;
	int i = d_cavebdryhead[threadId];
	d_cavebdryhead[threadId] = sindex;

	while (i != -1)
	{
		d_cavelist[sindex] = d_cavebdrylist[i];
		d_caveprev[sindex] = prev;
		d_cavenext[sindex] = -1;
		if (prev != -1)
			d_cavenext[prev] = sindex;
		d_cavethreadidx[sindex] = threadId;
		prev = sindex;
		sindex++;
		i = d_cavebdrynext[i];
	}

	d_cavebdrytail[threadId] = sindex - 1;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int tetexpandsize = 0;
	int bdryexpandsize = 0;

	int threadId = d_cavethreadidx[pos];

	REAL* insertpt = d_insertptlist + 3 * threadId;
	int cur = cavebdrycurstartindex + pos;
	tethandle cavetet = d_cavebdrylist[cur];
	tethandle neightet;
	cudamesh_fsym(cavetet, neightet, d_neighborlist);
	bool enqflag;
	REAL ori;

	if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
	{
		if (cudamesh_apex(cavetet, d_tetlist) != -1)
		{
			if (cudamesh_oppo(neightet, d_tetlist) != -1)
			{
				REAL *pts[3];
				int idx[3];
				idx[0] = cudamesh_org(cavetet, d_tetlist);
				idx[1] = cudamesh_dest(cavetet, d_tetlist);
				idx[2] = cudamesh_apex(cavetet, d_tetlist);
				for (int i = 0; i < 3; i++)
				{
					pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
				}
				ori = cuda_orient3d(pts[0], pts[1], pts[2], insertpt);
				enqflag = (ori > 0);
			}
			else
			{
				// It is a hull face. And its adjacent tet (at inside of the 
				//   domain) has been cut from the cavity. Cut it as well.
				enqflag = false;
			}
		}
		else
		{
			enqflag = true; // A hull edge
		}
		if (enqflag)
		{
			tetexpandsize = 1;
		}
		else
		{
			d_tetmarker[neightet.id] = MAXULL;
			d_cutcount[threadId] += 1; // This may cause a wrong value but it doesn't affect the result
			bdryexpandsize = 3;
		}
	}

	d_cavetetexpandsize[pos] = tetexpandsize;
	d_cavebdryexpandsize[pos] = bdryexpandsize;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int cur = cavebdrycurstartindex + pos;
	int threadId = d_cavebdryidx[cur];
	if (d_threadmarker[threadId] == -1)
		return;

	int bdryexpandsize = 0;

	REAL* insertpt = d_insertptlist + 3 * threadId;

	tethandle cavetet = d_cavebdrylist[cur];
	tethandle neightet;
	cudamesh_fsym(cavetet, neightet, d_neighborlist);
	bool enqflag;
	REAL ori;

	if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
	{
		if (cudamesh_apex(cavetet, d_tetlist) != -1)
		{
			if (cudamesh_oppo(neightet, d_tetlist) != -1)
			{
				REAL *pts[3];
				int idx[3];
				idx[0] = cudamesh_org(cavetet, d_tetlist);
				idx[1] = cudamesh_dest(cavetet, d_tetlist);
				idx[2] = cudamesh_apex(cavetet, d_tetlist);
				for (int i = 0; i < 3; i++)
				{
					pts[i] = cudamesh_id2pointlist(idx[i], d_pointlist);
				}
				ori = cuda_orient3d(pts[0], pts[1], pts[2], insertpt);
				enqflag = (ori > 0);
			}
			else
			{
				// It is a hull face. And its adjacent tet (at inside of the 
				//   domain) has been cut from the cavity. Cut it as well.
				enqflag = false;
			}
		}
		else
		{
			enqflag = true; // A hull edge
		}
		if(!enqflag)
		{
			d_tetmarker[neightet.id] = 0;
			bdryexpandsize = 3;
		}
	}
	d_cavebdryexpandsize[pos] = bdryexpandsize;
}

__global__ void kernelUpdateCavity2StarShapedSetThreadidx(
	int* d_cavethreadidx,
	int* d_cavetetexpandsize,
	int* d_cavetetexpandindices,
	int* d_cavetetthreadidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int* d_cavebdrythreadidx,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];

	int eindex;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		d_cavetetthreadidx[eindex] = threadId;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		for (int j = 0; j < 3; j++) {
			d_cavebdrythreadidx[eindex + j] = threadId;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];

	int cur = cavebdrycurstartindex + pos;
	tethandle cavetet = d_cavebdrylist[cur];

	int eindex, sindex, prev;
	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex;

		tethandle neightet, neineitet;
		cudamesh_fsym(cavetet, neightet, d_neighborlist);
		int newid;

		if (eindex == 0 || d_cavebdrythreadidx[eindex - 1] != threadId)
		{
			prev = d_cavebdrytail[threadId];
			d_cavebdrynext[prev] = sindex; // prev must not be -1
		}
		else
			prev = sindex - 1;

		// Add three new faces to find new boundaries.
		for (int j = 0; j < 3; j++)
		{
			newid = sindex + j;
			cudamesh_esym(neightet, neineitet);
			neineitet.ver = raw_epivot[neineitet.ver];
			d_cavebdrylist[newid] = neineitet;
			d_cavebdryprev[newid] = prev;
			d_cavebdrynext[newid] = newid + 1; // set to next one first
			prev = newid;
			cudamesh_enextself(neightet);
		}
		
		if (eindex + 2 == cavebdryexpandsize - 1 || d_cavebdrythreadidx[eindex + 3] != threadId)
		{
			//if (threadId == 153)
			//	printf("threadId = %d, cavebdryexpandsize = %d, eindex + 2 = %d, d_cavebdrythreadidx[eindex + 3] = %d\n",
			//		threadId, cavebdryexpandsize, eindex + 2, d_cavebdrythreadidx[eindex + 3]);
			d_cavebdrynext[newid] = -1;
		}
	}

	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex;

		if (eindex == 0 || d_cavetetthreadidx[eindex - 1] != threadId)
		{
			prev = d_cavetettail[threadId];
			if (prev != -1)
				d_cavetetnext[prev] = sindex;
			if (d_cavetethead[threadId] == -1) // initialize cavebdry list header
				d_cavetethead[threadId] = sindex;
		}
		else
			prev = sindex - 1;

		d_cavetetlist[sindex] = cavetet;
		d_cavetetprev[sindex] = prev;
		d_cavetetnext[sindex] = sindex + 1;

		if (eindex == cavetetexpandsize - 1 || d_cavetetthreadidx[eindex + 1] != threadId)
			d_cavetetnext[sindex] = -1;
	}
}

__global__ void kernelUpdateCavity2StarShapedAppend(
	tethandle* d_neighborlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	int* d_cavebdryexpandsize,
	int* d_cavebdryexpandindices,
	int cavebdrystartindex,
	int cavebdrycurstartindex,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_cavebdryexpandsize[pos] == 0)
		return;

	int cur = cavebdrycurstartindex + pos;
	int threadId = d_cavebdryidx[cur];

	tethandle cavetet = d_cavebdrylist[cur];

	int eindex = d_cavebdryexpandindices[pos];
	int sindex = cavebdrystartindex + eindex;

	tethandle neightet, neineitet;
	cudamesh_fsym(cavetet, neightet, d_neighborlist);
	int newid;

	// Add three new faces to find new boundaries.
	for (int j = 0; j < 3; j++)
	{
		newid = sindex + j;
		cudamesh_esym(neightet, neineitet);
		neineitet.ver = raw_epivot[neineitet.ver];
		d_cavebdrylist[newid] = neineitet;
		d_cavebdryidx[newid] = threadId;
		cudamesh_enextself(neightet);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];

	int sindex, eindex, prev;
	if (d_cavetetexpandsize[pos] != 0)
	{
		eindex = d_cavetetexpandindices[pos];
		sindex = cavetetstartindex + eindex;
		if (d_cavetetnext[sindex] == -1)
			d_cavetettail[threadId] = sindex;
	}

	if (d_cavebdryexpandsize[pos] != 0)
	{
		eindex = d_cavebdryexpandindices[pos];
		sindex = cavebdrystartindex + eindex + 2;
		if (d_cavebdrynext[sindex] == -1)
		{
			d_cavebdrytail[threadId] = sindex;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	tethandle cavetet, neightet, prevtet;

	if (d_cutcount[threadId] > 0)
	{
		// Reuse old space
		int cur = d_cavebdryhead[threadId];
		int prev = -1;
		int i = d_cavetethead[threadId];
		while (i != -1)
		{
			cavetet = d_cavetetlist[i];
			cudamesh_fsym(cavetet, neightet, d_neighborlist);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
			{
				d_cavebdrylist[cur] = cavetet;
				prev = cur;
				cur = d_cavebdrynext[cur];
			}
			i = d_cavetetnext[i];
			if (i == -1) // reach the end of new boundary faces
			{
				if (prev != -1)
				{
					d_cavebdrynext[prev] = -1;
					d_cavebdrytail[threadId] = prev;
				}
				else
				{
					// this should not happen
				}
			}
		}
	}
}

__global__ void kernelUpdateBoundaryFaces(
	tethandle* d_neighborlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	tethandle cavetet, neightet, prevtet;

	cavetet = d_cavebdrylist[pos];
	cudamesh_fsym(cavetet, neightet, d_neighborlist);
	if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) != threadId + 1)
	{
		d_cavebdryidx[pos] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	tethandle cavetet, neightet;

	if (d_cutcount[threadId] > 0)
	{
		// Reuse old space
		int prev = -1;
		int i = d_caveoldtethead[threadId];
		while (i != -1)
		{
			cavetet = d_caveoldtetlist[i];
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[cavetet.id]) == threadId)
			{
				if (prev != -1)
					d_caveoldtetnext[prev] = i;
				else
					d_caveoldtethead[threadId] = i;
				d_caveoldtetprev[i] = prev;
				prev = i;
			}
			i = d_caveoldtetnext[i];
			if (i == -1) // reach the end of new boundary faces
			{
				if (prev != -1)
				{
					d_caveoldtetnext[prev] = -1;
					d_caveoldtettail[threadId] = prev;
				}
				else
				{
					// The cavity should contain at least one tet
					// Usually this would not happen
					int eleidx = d_insertidxlist[threadId];
					if (threadmarker == 0)
						d_segstatus[eleidx].setAbortive(true);
					else if (threadmarker == 1)
						d_tristatus[eleidx].setAbortive(true);
					else if (threadmarker == 2)
						d_tetstatus[eleidx].setAbortive(true);
					d_threadmarker[threadId] = -1;
				}
			}
		}
	}
}

__global__ void kernelUpdateOldTets(
	tethandle* d_neighborlist,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	uint64* d_tetmarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == -1)
		return;

	tethandle cavetet, neightet;

	cavetet = d_caveoldtetlist[pos];
	if (cudamesh_getUInt64PriorityIndex(d_tetmarker[cavetet.id]) != threadId + 1)
	{
		d_caveoldtetidx[pos] = -1;
	}
}


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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	int neighborId;
	tethandle cavetet;
	
	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		cavetet = d_cavebdrylist[i];
		neighborId = cudamesh_getUInt64PriorityIndex(d_tetmarker[cavetet.id]);
		if (neighborId != MAXUINT && neighborId != threadId) // neighbor also marked
		{
			if (d_threadmarker[neighborId] != -1) // neighbor is alive also
			{
				if(threadId > neighborId)
				{
					d_threadmarker[threadId] = -1;
					return;
				}
			}
		}
		i = d_cavebdrynext[i];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	trihandle parysh;
	tethandle neightet;

	// Reuse old space
	bool enqflag;
	int cutshcount = 0;
	int prev = -1;
	int i = d_caveshhead[threadId]; // for dangling segment, this is -1
	while (i != -1)
	{
		parysh = d_caveshlist[i];
		if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) == threadId)
		{
			enqflag = false;
			cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
			{
				cudamesh_fsymself(neightet, d_neighborlist);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId)
					enqflag = true;
			}
			if (enqflag)
			{
				if (prev != -1)
					d_caveshnext[prev] = i;
				else
					d_caveshhead[threadId] = i;
				d_caveshprev[i] = prev;
				prev = i;
			}
			else
			{
				d_trimarker[parysh.id] = MAXULL;
				cutshcount++;
			}
		}
		i = d_caveshnext[i];
		if (i == -1) // reach the end of subcavity faces
		{
			if (prev != -1)
			{
				d_caveshnext[prev] = -1;
				d_caveshtail[threadId] = prev;
			}
		}
	}
	d_cutshcount[pos] = cutshcount;
}

__global__ void kernelUpdateSubcavities(
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	trihandle parysh;
	tethandle neightet;

	bool enqflag;
	parysh = d_caveshlist[pos];
	if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) == threadId + 1)
	{
		enqflag = false;
		cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
		{
			cudamesh_fsymself(neightet, d_neighborlist);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) == threadId + 1)
				enqflag = true;
		}
		if (!enqflag)
		{
			d_trimarker[parysh.id] = 0;
			d_caveshidx[pos] = -1;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int cutshcount = d_cutshcount[pos];
	if (cutshcount == 0)
		return;

	int threadmarker = d_threadmarker[threadId];
	locateresult loc = d_pointlocation[threadId];
	int i = 0;
	trihandle splitsh, neighsh;

	if (loc == ONFACE)
	{
		if (threadmarker == 1)
		{
			splitsh = d_searchsh[threadId];
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[splitsh.id]) != threadId)
			{
				//printf("threadId #%d - Invalid trimarker #%d - %d\n", threadId, splitsh.id, cudamesh_getUInt64PriorityIndex(d_trimarker[splitsh.id]));
				i++;
			}
		}
	}
	else if (loc == ONEDGE)
	{
		if (threadmarker == 0)
		{
			int segId = d_insertidxlist[threadId];
			trihandle splitseg(segId, 0);
			if (d_segmarker[segId] != threadId)
			{
				//printf("threadId #%d - Invalid segmarker %d\n", threadId, d_segmarker[segId]);
				i++;
			}

			cudamesh_spivot(splitseg, splitsh, d_seg2trilist);
		}
		else if (threadmarker == 1)
		{
			splitsh = d_searchsh[threadId];
		}

		if (splitsh.id != -1)
		{
			// All subfaces at this edge should be in subcavity
			int pa = cudamesh_sorg(splitsh, d_trifacelist);
			neighsh = splitsh;
			while (1)
			{
				if (cudamesh_sorg(neighsh, d_trifacelist) != pa)
				{
					cudamesh_sesymself(neighsh);
				}
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId)
				{
					//printf("threadId #%d - Invalid trimarker #%d - %d\n", threadId, neighsh.id, cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]));
					i++;
				}
				cudamesh_spivotself(neighsh, d_tri2trilist);
				if (neighsh.id == splitsh.id) break;
				if (neighsh.id == -1) break;
			}
		}
	}

	if (i > 0)
	{
		int eleidx = d_insertidxlist[threadId];
		if (threadmarker == 0)
		{
			d_segstatus[eleidx].setAbortive(true);
		}
		else if (threadmarker == 1)
		{
			d_tristatus[eleidx].setAbortive(true);
		}
		d_threadmarker[threadId] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == -1)
		return;

	locateresult loc = d_pointlocation[threadId];
	int i = 0;
	trihandle splitsh, neighsh;

	if (loc == ONFACE)
	{
		if (threadmarker == 1)
		{
			splitsh = d_searchsh[threadId];
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[splitsh.id]) != threadId + 1)
			{
				i++;
			}
		}
	}
	else if (loc == ONEDGE)
	{
		if (threadmarker == 0)
		{
			int segId = d_insertidxlist[threadId];
			trihandle splitseg(segId, 0);
			if (d_segmarker[segId] != threadId + 1)
			{
				i++;
			}

			cudamesh_spivot(splitseg, splitsh, d_seg2trilist);
		}
		else if (threadmarker == 1)
		{
			splitsh = d_searchsh[threadId];
		}

		if (splitsh.id != -1)
		{
			// All subfaces at this edge should be in subcavity
			int pa = cudamesh_sorg(splitsh, d_trifacelist);
			neighsh = splitsh;
			while (1)
			{
				if (cudamesh_sorg(neighsh, d_trifacelist) != pa)
				{
					cudamesh_sesymself(neighsh);
				}
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId + 1)
				{
					i++;
				}
				cudamesh_spivotself(neighsh, d_tri2trilist);
				if (neighsh.id == splitsh.id) break;
				if (neighsh.id == -1) break;
			}
		}
	}

	if (i > 0)
	{
		int eleidx = d_insertidxlist[threadId];
		if (threadmarker == 0)
		{
			d_segstatus[eleidx].setAbortive(true);
		}
		else if (threadmarker == 1)
		{
			d_tristatus[eleidx].setAbortive(true);
		}
		d_threadmarker[threadId] = -1;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	int insertidx = d_insertidxlist[threadId];
	if (threadmarker == 0)
	{
		tethandle spintet;
		tethandle searchtet = d_searchtet[threadId];
		spintet = searchtet;
		while (1) 
		{
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId)
			{
				d_segstatus[insertidx].setAbortive(true);
				d_threadmarker[threadId] = -1;
				break;
			}
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id)
				break;
		}
	}
	else if (threadmarker == 1)
	{
		int elementid = d_searchsh[threadId].id;
		if (cudamesh_getUInt64PriorityIndex(d_trimarker[elementid]) != threadId)
		{
			d_tristatus[insertidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
	else if (threadmarker == 2)
	{
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[insertidx]) != threadId)
		{
			d_tetstatus[insertidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == -1)
		return;

	int insertidx = d_insertidxlist[threadId];
	if (threadmarker == 0)
	{
		tethandle spintet;
		tethandle searchtet = d_searchtet[threadId];
		spintet = searchtet;
		while (1)
		{
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId + 1)
			{
				d_segstatus[insertidx].setAbortive(true);
				d_threadmarker[threadId] = -1;
				break;
			}
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id)
				break;
		}
	}
	else if (threadmarker == 1)
	{
		int elementid = d_searchsh[threadId].id;
		if (cudamesh_getUInt64PriorityIndex(d_trimarker[elementid]) != threadId + 1)
		{
			d_tristatus[insertidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
	else if (threadmarker == 2)
	{
		if (cudamesh_getUInt64PriorityIndex(d_tetmarker[insertidx]) != threadId + 1)
		{
			d_tetstatus[insertidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int insertiontype = d_threadmarker[threadId];
	int insertidx = d_insertidxlist[threadId];
	REAL* insertpt = d_insertptlist + 3 * threadId;
	tethandle searchtet, spintet;
	searchtet = d_searchtet[threadId];
	int ptidx, i;
	REAL* pt, rd;
	REAL minedgelength = raw_kernelconstants[0];
	locateresult loc = d_pointlocation[threadId];
	if (loc == ONEDGE)
	{
		spintet = searchtet;
		ptidx = cudamesh_org(spintet, d_tetlist);
		pt = cudamesh_id2pointlist(ptidx, d_pointlist);
		rd = cudamesh_distance(pt, insertpt);
		if (rd < minedgelength)
		{
			if (insertiontype == 0)
				d_segstatus[insertidx].setAbortive(true);
			else if (insertiontype == 1)
				d_tristatus[insertidx].setAbortive(true);
			else if (insertiontype == 2)
				d_tetstatus[insertidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
			return;
		}
		ptidx = cudamesh_dest(spintet, d_tetlist);
		pt = cudamesh_id2pointlist(ptidx, d_pointlist);
		rd = cudamesh_distance(pt, insertpt);
		if (rd < minedgelength)
		{
			if (insertiontype == 0)
				d_segstatus[insertidx].setAbortive(true);
			else if (insertiontype == 1)
				d_tristatus[insertidx].setAbortive(true);
			else if (insertiontype == 2)
				d_tetstatus[insertidx].setAbortive(true);
			d_threadmarker[threadId] = -1;
			return;
		}

		while (1) {
			ptidx = cudamesh_apex(spintet, d_tetlist);
			if (ptidx != -1)
			{
				pt = cudamesh_id2pointlist(ptidx, d_pointlist);
				rd = cudamesh_distance(pt, insertpt);
				if (rd < minedgelength)
				{
					if (insertiontype == 0)
						d_segstatus[insertidx].setAbortive(true);
					else if (insertiontype == 1)
						d_tristatus[insertidx].setAbortive(true);
					else if (insertiontype == 2)
						d_tetstatus[insertidx].setAbortive(true);
					d_threadmarker[threadId] = -1;
					return;
				}
			}
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == searchtet.id)
				break;
		}
	}
	else if (loc == ONFACE)
	{
		for (i = 0; i < 3; i++)
		{
			ptidx = d_tetlist[4 * searchtet.id + i];
			pt = cudamesh_id2pointlist(ptidx, d_pointlist);
			rd = cudamesh_distance(pt, insertpt);
			if (rd < minedgelength)
			{
				if (insertiontype == 1)
					d_tristatus[insertidx].setAbortive(true);
				else if (insertiontype == 2)
					d_tetstatus[insertidx].setAbortive(true);
				d_threadmarker[threadId] = -1;
				return;
			}
		}
		ptidx = d_tetlist[4 * searchtet.id + 3];
		if (ptidx != -1)
		{
			pt = cudamesh_id2pointlist(ptidx, d_pointlist);
			rd = cudamesh_distance(pt, insertpt);
			if (rd < minedgelength)
			{
				if (insertiontype == 1)
					d_tristatus[insertidx].setAbortive(true);
				else if (insertiontype == 2)
					d_tetstatus[insertidx].setAbortive(true);
				d_threadmarker[threadId] = -1;
				return;
			}
		}
		cudamesh_fsym(searchtet, spintet, d_neighborlist);
		ptidx = cudamesh_oppo(spintet, d_tetlist);
		if (ptidx != -1)
		{
			pt = cudamesh_id2pointlist(ptidx, d_pointlist);
			rd = cudamesh_distance(pt, insertpt);
			if (rd < minedgelength)
			{
				if (insertiontype == 1)
					d_tristatus[insertidx].setAbortive(true);
				else if (insertiontype == 2)
					d_tetstatus[insertidx].setAbortive(true);
				d_threadmarker[threadId] = -1;
				return;
			}
		}
	}
	else if (loc == INTETRAHEDRON)
	{
		for (i = 0; i < 4; i++)
		{
			ptidx = d_tetlist[4 * searchtet.id + i];
			pt = cudamesh_id2pointlist(ptidx, d_pointlist);
			rd = cudamesh_distance(pt, insertpt);
			if (rd < minedgelength)
			{
				if (insertiontype == 2)
					d_tetstatus[insertidx].setAbortive(true);
				d_threadmarker[threadId] = -1;
				return;
			}
		}
	}
}

__global__ void kernelResetCavityReuse(
	int* d_insertidxlist,
	int* d_threadlist,
	tristatus* d_segstatus,
	tristatus* d_tristatus,
	tetstatus* d_tetstatus,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int eleidx = d_insertidxlist[threadId];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 0)
	{
		if (d_segstatus[eleidx].isCavityReuse())
			d_segstatus[eleidx].setCavityReuse(false);
	}
	else if (threadmarker == 1)
	{
		if (d_tristatus[eleidx].isCavityReuse())
			d_tristatus[eleidx].setCavityReuse(false);
	}
	else if (threadmarker == 2)
	{
		if (d_tetstatus[eleidx].isCavityReuse())
			d_tetstatus[eleidx].setCavityReuse(false);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2) // a tetrahedron
		return;

	REAL* insertpt = d_insertptlist + 3 * threadId;

	tethandle cavetet;
	int ptidx, parentpt;
	REAL *pts, smlen = -1.0, len;

	int i = d_caveoldtethead[threadId], j;
	cavetet = d_caveoldtetlist[i];
	ptidx = d_tetlist[4 * cavetet.id + 0];
	pts = cudamesh_id2pointlist(ptidx, d_pointlist);
	smlen = cudamesh_distance(pts, insertpt);
	parentpt = ptidx;

	while (i != -1)
	{
		cavetet = d_caveoldtetlist[i];
		for (j = 0; j < 4; j++)
		{
			ptidx = d_tetlist[4 * cavetet.id + j];
			if (ptidx == -1)
				continue;
			pts = cudamesh_id2pointlist(ptidx, d_pointlist);
			len = cudamesh_distance(pts, insertpt);
			if(len < smlen)
			{
				smlen = len;
				parentpt = ptidx;
			}
		}
		i = d_caveoldtetnext[i];
	}

	d_smlen[threadId] = smlen;
	d_parentpt[threadId] = parentpt;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	tethandle spintet, neightet, neineitet;
	trihandle paryseg, checkseg;

	// Reuse old space
	int j, k, markeridx;
	int prev = -1;
	int i = d_cavetetseghead[threadId];
	while (i != -1)
	{
		paryseg = d_cavetetseglist[i];
		if (d_segmarker[paryseg.id] != threadId) // not a splitting segment
		{
			// Check if the segment is inside the cavity.
			//   'j' counts the num of adjacent tets of this seg.
			//   'k' counts the num of adjacent tets which are 'infected'.
			j = k = 0;
			cudamesh_sstpivot1(paryseg, neightet, d_seg2tetlist);
			spintet = neightet;
			while (1) {
				j++;
				markeridx = cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]);
				if (markeridx != threadId) // outside cavity
				{
					// Remember it only when it is not inside other cavities
					// (possible when cavities share edges/segments)
					if (markeridx == MAXUINT || (markeridx != MAXUINT && d_threadmarker[markeridx] == -1)) // a unmarked tet or a tet belongs to loser
						neineitet = spintet;
				}
				else
				{
					k++;
				}
				cudamesh_fnextself(spintet, d_neighborlist);
				if (spintet.id == neightet.id)
					break;
			}
			if (k == 0) // should be removed
			{

			}
			else if (k < j) // on the boundary
			{
				assert(neineitet.id != -1); // there must be a tet that is not included in any cavities
				// connect it to the recorded outer tet
				cudamesh_sstbond1(paryseg, neineitet, d_seg2tetlist);
				// update cavetetseg list
				if (prev != -1)
					d_cavetetsegnext[prev] = i;
				else
					d_cavetetseghead[threadId] = i;
				d_cavetetsegprev[i] = prev;
				prev = i;
			}
			else // impossible
			{
				assert(0);
				printf("Error: Segment #%d is inside the cavity!\n", paryseg.id);
			}
		}
		i = d_cavetetsegnext[i];
		if (i == -1) // reach the end of cavetetseg
		{
			if (prev != -1) // when there is at least one boundary segment
			{
				d_cavetetsegnext[prev] = -1;
				d_cavetetsegtail[threadId] = prev;
			}
			else // no boundary segment
			{
				d_cavetetseghead[threadId] = -1;
				d_cavetetsegtail[threadId] = -1;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetsegidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	tethandle spintet, neightet, neineitet;
	trihandle paryseg, checkseg;

	// Reuse old space
	int j, k, markeridx;

	paryseg = d_cavetetseglist[pos];
	if (d_segmarker[paryseg.id] != threadId + 1) // not a splitting segment
	{
		// Check if the segment is inside the cavity.
		//   'j' counts the num of adjacent tets of this seg.
		//   'k' counts the num of adjacent tets which are 'infected'.
		j = k = 0;
		cudamesh_sstpivot1(paryseg, neightet, d_seg2tetlist);
		spintet = neightet;
		while (1) {
			j++;
			markeridx = cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]);
			if (markeridx != threadId + 1) // outside cavity
			{
				// Remember it only when it is not inside other cavities
				// (possible when cavities share edges/segments)
				if (markeridx == 0 || (markeridx != 0 && d_threadmarker[markeridx - 1] == -1)) // a unmarked tet or a tet belongs to loser
					neineitet = spintet;
			}
			else
			{
				k++;
			}
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == neightet.id)
				break;
		}
		if (k == 0) // should be removed
		{
			d_cavetetsegidx[pos] = -1;
		}
		else if (k < j) // on the boundary
		{
			// connect it to the recorded outer tet
			cudamesh_sstbond1(paryseg, neineitet, d_seg2tetlist);
		}
		else // impossible
		{
			printf("Error: Segment #%d is inside the cavity!\n", paryseg.id);
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	tethandle neightet;
	trihandle parysh, checksh;

	// Reuse old space
	int j, k;
	int prev = -1;
	int i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		parysh = d_cavetetshlist[i];
		if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) != threadId) // not inside subcavity
		{
			// Check if this subface is inside the cavity.
			k = 0;
			for (j = 0; j < 2; j++)
			{
				cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) != threadId)
				{
					checksh = parysh; // remember this side
				}
				else
				{
					k++;
				}
				cudamesh_sesymself(parysh);
			}

			if (k == 0) // should be removed
			{

			}
			else if (k == 1) // on the boundary
			{
				parysh = checksh;
				// update cavetetsh list
				if (prev != -1)
					d_cavetetshnext[prev] = i;
				else
					d_cavetetshhead[threadId] = i;
				d_cavetetshprev[i] = prev;
				d_cavetetshlist[i] = parysh;
				prev = i;
			}
			else // impossible
			{
				assert(0);
				printf("Error: Subface #%d is inside the cavity!\n", parysh.id);
			}
		}
		i = d_cavetetshnext[i];
		if (i == -1) // reach the end of cavetetsh
		{
			if (prev != -1) // when there is at least one boundary subface
			{
				d_cavetetshnext[prev] = -1;
				d_cavetetshtail[threadId] = prev;
			}
			else // no boundary subface
			{
				d_cavetetshhead[threadId] = -1;
				d_cavetetshtail[threadId] = -1;
			}
		}
	}
}

__global__ void kernelUpdateCavitySubfaces(
	tethandle* d_neighborlist,
	tethandle* d_tri2tetlist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	uint64* d_tetmarker,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	if (d_threadmarker[threadId] == -1)
		return;

	tethandle neightet;
	trihandle parysh, checksh;

	// Reuse old space
	int j, k;
	parysh = d_cavetetshlist[pos];

	if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) != threadId + 1) // not inside subcavity
	{
		// Check if this subface is inside the cavity.
		k = 0;
		for (j = 0; j < 2; j++)
		{
			cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[neightet.id]) != threadId + 1)
			{
				checksh = parysh; // remember this side
			}
			else
			{
				k++;
			}
			cudamesh_sesymself(parysh);
		}

		if (k == 0) // should be removed
		{
			d_cavetetshidx[pos] = -1;
		}
		else if (k == 1) // on the boundary
		{
			parysh = checksh;
			d_cavetetshlist[pos] = parysh;
		}
		else // impossible
		{
			printf("Error: Subface #%d is inside the cavity!\n", parysh.id);
		}
	}
}

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	int newidx = oldpointsize + pos;
	if (threadmarker == 0)
		d_pointtypelist[newidx] = FREESEGVERTEX;
	else if(threadmarker == 1)
		d_pointtypelist[newidx] = FREEFACETVERTEX;
	else
		d_pointtypelist[newidx] = FREEVOLVERTEX;

	newidx *= 3;
	REAL* insertpt = d_insertptlist + 3 * threadId;
	d_pointlist[newidx++] = insertpt[0];
	d_pointlist[newidx++] = insertpt[1];
	d_pointlist[newidx++] = insertpt[2];
}

__global__ void kernelInsertNewPoints(
	int* d_threadlist,
	REAL* d_pointlist,
	verttype* d_pointtypelist,
	REAL* d_insertptlist,
	int* d_threadmarker,
	int* d_threadpos,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	d_threadpos[threadId] = pos;

	int newidx = oldpointsize + pos;
	if (threadmarker == 0)
		d_pointtypelist[newidx] = FREESEGVERTEX;
	else if (threadmarker == 1)
		d_pointtypelist[newidx] = FREEFACETVERTEX;
	else
		d_pointtypelist[newidx] = FREEVOLVERTEX;

	newidx *= 3;
	REAL* insertpt = d_insertptlist + 3 * threadId;
	d_pointlist[newidx++] = insertpt[0];
	d_pointlist[newidx++] = insertpt[1];
	d_pointlist[newidx++] = insertpt[2];
}

__global__ void kernelSetCavityThreadIdx(
	int* d_cavethreadidx,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavethreadidx[pos];
	if (threadId != -1 && d_threadmarker[threadId] == -1)
		d_cavethreadidx[pos] = -1;
}

__global__ void kernelComputeShortestEdgeLength_Phase1(
	int* d_cavebdryidx,
	int* d_scanleft,
	int* d_scanright,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2) // a tetrahedron
		return;

	if (pos == 0 || d_cavebdryidx[pos - 1] != threadId)
		d_scanleft[threadId] = pos;
	if (pos == numofthreads - 1 || d_cavebdryidx[pos + 1] != threadId)
		d_scanright[threadId] = pos;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2) // a tetrahedron
		return;

	REAL* insertpt = d_insertptlist + 3 * threadId;

	tethandle cavetet;
	int ptidx, parentpt;
	REAL *pts, smlen = -1.0, len;

	int left, right;
	left = d_scanleft[threadId];
	right = d_scanright[threadId];

	int ipt[3];

	int i = left, j;
	cavetet = d_cavebdrylist[i];
	ptidx = cudamesh_org(cavetet, d_tetlist);
	pts = cudamesh_id2pointlist(ptidx, d_pointlist);
	smlen = cudamesh_distance(pts, insertpt);
	parentpt = ptidx;

	for(i = left; i<= right; i++)
	{
		cavetet = d_cavebdrylist[i];
		ipt[0] = cudamesh_org(cavetet, d_tetlist);
		ipt[1] = cudamesh_apex(cavetet, d_tetlist);
		ipt[2] = cudamesh_dest(cavetet, d_tetlist);
		for (j = 0; j < 3; j++)
		{
			ptidx = ipt[j];
			if (ptidx == -1)
				continue;
			pts = cudamesh_id2pointlist(ptidx, d_pointlist);
			len = cudamesh_distance(pts, insertpt);
			if (len < smlen)
			{
				smlen = len;
				parentpt = ptidx;
			}
		}
	}

	d_smlen[threadId] = smlen;
	d_parentpt[threadId] = parentpt;
}

__global__ void kernelCountNewTets(
	int* d_threadlist,
	tethandle* d_cavebdrylist,
	int* d_cavebdrynext,
	int* d_cavebdryhead,
	int* d_tetexpandsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int expandsize = 0;
	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		expandsize++;
		i = d_cavebdrynext[i];
	}
	d_tetexpandsize[pos] = expandsize;
}

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
	int* d_tetexpandindices,
	int* d_emptytetindices,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int startidx = d_tetexpandindices[pos], newtetidx;
	int newptidx = oldpointsize + pos;
	tethandle neightet, oldtet, newtet;

	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		newtetidx = d_emptytetindices[startidx++];
		neightet = d_cavebdrylist[i];
		cudamesh_fsym(neightet, oldtet, d_neighborlist); // Get the oldtet (inside the cavity).

		// There might be duplicate elements in cavebdrylist.
		// In that case, oldtet will be newtet. Check to avoid
		if (!d_tetstatus[oldtet.id].isEmpty())
		{
			if (cudamesh_apex(neightet, d_tetlist) != -1)
			{
				// Create a new tet in the cavity
				newtet.id = newtetidx;
				newtet.ver = 11;
				cudamesh_setorg(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
				cudamesh_setdest(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
				cudamesh_setapex(newtet, cudamesh_apex(neightet, d_tetlist), d_tetlist);
				cudamesh_setoppo(newtet, newptidx, d_tetlist);
			}
			else
			{
				// Create a new hull tet
				newtet.id = newtetidx;
				newtet.ver = 11;
				cudamesh_setorg(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
				cudamesh_setdest(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
				cudamesh_setapex(newtet, newptidx, d_tetlist);
				cudamesh_setoppo(newtet, -1, d_tetlist); // It must opposite to face 3.
				// Adjust back to the cavity bounday face.
				cudamesh_esymself(newtet);
			}
			// Connect newtet <==> neightet, this also disconnect the old bond.
			cudamesh_bond(newtet, neightet, d_neighborlist);
			// Oldtet still connects to neightet
			d_cavebdrylist[i] = oldtet;
		}
		else // duplicate elements cause fake oldtet
		{
			d_cavebdrylist[i] = tethandle(-1, 11);
		}

		i = d_cavebdrynext[i];
	}

	d_point2tetlist[newptidx] = newtet;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	int newtetidx = d_emptytetindices[pos];
	int newptidx = oldpointsize + d_threadpos[threadId];
	tethandle neightet, oldtet, newtet;

	neightet = d_cavebdrylist[pos];
	cudamesh_fsym(neightet, oldtet, d_neighborlist); // Get the oldtet (inside the cavity).

	if (cudamesh_apex(neightet, d_tetlist) != -1)
	{
		// Create a new tet in the cavity
		newtet.id = newtetidx;
		newtet.ver = 11;
		cudamesh_setorg(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
		cudamesh_setdest(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
		cudamesh_setapex(newtet, cudamesh_apex(neightet, d_tetlist), d_tetlist);
		cudamesh_setoppo(newtet, newptidx, d_tetlist);
	}
	else
	{
		// Create a new hull tet
		newtet.id = newtetidx;
		newtet.ver = 11;
		cudamesh_setorg(newtet, cudamesh_org(neightet, d_tetlist), d_tetlist);
		cudamesh_setdest(newtet, cudamesh_dest(neightet, d_tetlist), d_tetlist);
		cudamesh_setapex(newtet, newptidx, d_tetlist);
		cudamesh_setoppo(newtet, -1, d_tetlist); // It must opposite to face 3.
		// Adjust back to the cavity bounday face.
		cudamesh_esymself(newtet);
	}
	// Connect newtet <==> neightet, this also disconnect the old bond.
	cudamesh_bond(newtet, neightet, d_neighborlist);
	// Oldtet still connects to neightet
	d_cavebdrylist[pos] = oldtet;

	d_point2tetlist[newptidx] = newtet;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	tethandle oldtet, neightet, newtet, newneitet, spintet;
	int orgidx;

	int i = d_cavebdryhead[threadId], j;

	while (i != -1)
	{
		// Get the newtet and oldtet at the same face.
		oldtet = d_cavebdrylist[i];
		if (oldtet.id != -1) // not fake one
		{
			cudamesh_fsym(oldtet, neightet, d_neighborlist);
			cudamesh_fsym(neightet, newtet, d_neighborlist);

			// Comment: oldtet and newtet must be at the same directed edge.
			// Connect the three other faces of this newtet.
			for (j = 0; j < 3; j++)
			{
				cudamesh_esym(newtet, neightet); // Go to the face
				// Do not have neighbor yet
				if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id == -1)
				{
					// Find the adjacent face of this new tet
					spintet = oldtet;
					while (1)
					{
						cudamesh_fnextself(spintet, d_neighborlist);
						if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId)
							break;
					}
					cudamesh_fsym(spintet, newneitet, d_neighborlist);
					cudamesh_esymself(newneitet);
					cudamesh_bond(neightet, newneitet, d_neighborlist);
				}
				orgidx = cudamesh_org(newtet, d_tetlist);
				if(orgidx != -1)
					d_point2tetlist[orgidx] = newtet;
				cudamesh_enextself(newtet);
				cudamesh_enextself(oldtet);
			}
			d_cavebdrylist[i] = newtet; // Save the new tet

			// Update tetstatus
			d_tetstatus[oldtet.id].clear();
			d_tetstatus[newtet.id].setEmpty(false);
		}

		i = d_cavebdrynext[i];
	}

	// Check neighbor
	//i = d_cavebdryhead[threadId];
	//while (i != -1)
	//{
	//	newtet = d_cavebdrylist[i];
	//	if (newtet.id != -1)
	//	{
	//		for (j = 0; j < 4; j++)
	//		{
	//			newtet.ver = j;
	//			neightet = d_neighborlist[4 * newtet.id + (newtet.ver & 3)];
	//			if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id != newtet.id)
	//				printf("Wrong neighbor(%d): Tet#%d - %d, %d, %d, %d, Tet#%d - %d, %d, %d, %d\n",
	//					threadId,
	//					newtet.id,
	//					d_neighborlist[4 * newtet.id + 0].id, d_neighborlist[4 * newtet.id + 1].id,
	//					d_neighborlist[4 * newtet.id + 2].id, d_neighborlist[4 * newtet.id + 3].id,
	//					neightet.id,
	//					d_neighborlist[4 * neightet.id + 0].id, d_neighborlist[4 * neightet.id + 1].id,
	//					d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
	//		}
	//	}
	//	i = d_cavebdrynext[i];
	//}
}

__global__ void kernelConnectNewTetNeighbors(
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	tethandle* d_point2tetlist,
	int* d_tetlist,
	tethandle* d_neighborlist,
	tetstatus* d_tetstatus,
	uint64* d_tetmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	tethandle oldtet, neightet, newtet, newneitet, spintet;
	int orgidx, j;

	// Get the newtet and oldtet at the same face.
	oldtet = d_cavebdrylist[pos];
	cudamesh_fsym(oldtet, neightet, d_neighborlist);
	cudamesh_fsym(neightet, newtet, d_neighborlist);

	// Comment: oldtet and newtet must be at the same directed edge.
	// Connect the three other faces of this newtet.
	for (j = 0; j < 3; j++)
	{
		cudamesh_esym(newtet, neightet); // Go to the face
		// Do not have neighbor yet
		if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id == -1)
		{
			// Find the adjacent face of this new tet
			spintet = oldtet;
			while (1)
			{
				cudamesh_fnextself(spintet, d_neighborlist);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId + 1)
					break;
			}
			cudamesh_fsym(spintet, newneitet, d_neighborlist);
			cudamesh_esymself(newneitet);
			cudamesh_bond(neightet, newneitet, d_neighborlist);
		}
		orgidx = cudamesh_org(newtet, d_tetlist);
		if (orgidx != -1)
			d_point2tetlist[orgidx] = newtet;
		cudamesh_enextself(newtet);
		cudamesh_enextself(oldtet);
	}
	d_cavebdrylist[pos] = newtet; // Save the new tet

	// Update tetstatus
	d_tetstatus[oldtet.id].clear();
	d_tetstatus[newtet.id].setEmpty(false);

	// Check neighbor
	newtet = d_cavebdrylist[pos];
	for (j = 0; j < 4; j++)
	{
		newtet.ver = j;
		neightet = d_neighborlist[4 * newtet.id + (newtet.ver & 3)];
		if (d_neighborlist[4 * neightet.id + (neightet.ver & 3)].id != newtet.id)
		printf("Wrong neighbor(%d): Tet#%d - %d, %d, %d, %d, Tet#%d - %d, %d, %d, %d\n",
		threadId,
		newtet.id,
		d_neighborlist[4 * newtet.id + 0].id, d_neighborlist[4 * newtet.id + 1].id,
		d_neighborlist[4 * newtet.id + 2].id, d_neighborlist[4 * newtet.id + 3].id,
		neightet.id,
		d_neighborlist[4 * neightet.id + 0].id, d_neighborlist[4 * neightet.id + 1].id,
		d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	trihandle parysh;
	tethandle neightet, newtet;
	int i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		parysh = d_cavetetshlist[i]; // this is connect to a outside tet
		// Connect it if it is a boundary subface
		if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) != threadId)
		{
			cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
			cudamesh_fsym(neightet, newtet, d_neighborlist);
			cudamesh_sesymself(parysh);
			cudamesh_tsbond(newtet, parysh, d_tet2trilist, d_tri2tetlist);
		}
		i = d_cavetetshnext[i];
	}
}

__global__ void kernelConnectBoundarySubfaces2NewTets(
	tethandle* d_tri2tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_cavetetshlist,
	int* d_cavetetshidx,
	uint64* d_trimarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	trihandle parysh;
	tethandle neightet, newtet;

	parysh = d_cavetetshlist[pos]; // this is connect to a outside tet
	// Connect it if it is a boundary subface
	if (cudamesh_getUInt64PriorityIndex(d_trimarker[parysh.id]) != threadId + 1)
	{
		cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
		cudamesh_fsym(neightet, newtet, d_neighborlist);
		cudamesh_sesymself(parysh);
		cudamesh_tsbond(newtet, parysh, d_tet2trilist, d_tri2tetlist);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	trihandle paryseg;
	tethandle neightet, spintet;
	int i = d_cavetetseghead[threadId];
	while (i != -1)
	{
		paryseg = d_cavetetseglist[i];
		// Connect it if it is a boundary subseg
		if (d_segmarker[paryseg.id] != threadId)
		{
			cudamesh_sstpivot1(paryseg, neightet, d_seg2tetlist);
			spintet = neightet;
			while (1)
			{
				cudamesh_tssbond1(spintet, paryseg, d_tet2seglist);
				cudamesh_fnextself(spintet, d_neighborlist);
				if (spintet.id == neightet.id)
					break;
			}
		}
		else
		{
			// This may happen when there is only one splitting segment
		}
		i = d_cavetetsegnext[i];
	}
}

__global__ void kernelConnectBoundarySubsegs2NewTets(
	tethandle* d_seg2tetlist,
	tethandle* d_neighborlist,
	trihandle* d_tet2seglist,
	trihandle* d_cavetetseglist,
	int* d_cavetetsegidx,
	int* d_segmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetsegidx[pos];
	trihandle paryseg;
	tethandle neightet, spintet;

	paryseg = d_cavetetseglist[pos];
	// Connect it if it is a boundary subseg
	if (d_segmarker[paryseg.id] != threadId + 1)
	{
		cudamesh_sstpivot1(paryseg, neightet, d_seg2tetlist);
		spintet = neightet;
		while (1)
		{
			cudamesh_tssbond1(spintet, paryseg, d_tet2seglist);
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == neightet.id)
				break;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
	{
		d_caveshbdsize[pos] = 0;
		return;
	}

	trihandle cavesh, neighsh;
	REAL sign;

	int caveshbdsize = 0;
	int i = d_caveshhead[threadId], j;
	while (i != -1)
	{
		cavesh = d_caveshlist[i];
		for (j = 0; j < 3; j++)
		{
			if (!cudamesh_isshsubseg(cavesh, d_tri2seglist))
			{
				cudamesh_spivot(cavesh, neighsh, d_tri2trilist);
				if (neighsh.id != -1)
				{
					if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId)
					{
						// A boundary edge
						sign = 1;
					}
					else
					{
						// Internal edge
						sign = -1;
					}
				}
				else
				{
					// A boundary edge
					sign = 1;
				}
			}
			else
			{
				// A segment. It is a boundary edge
				sign = 1;
			}
			if (sign >= 0)
			{
				caveshbdsize++;
			}
			cudamesh_senextself(cavesh);
		}

		i = d_caveshnext[i];
	}
	d_caveshbdsize[pos] = caveshbdsize;
}

__global__ void kernelSubCavityBoundaryEdgeCheck(
	trihandle* d_tri2seglist,
	trihandle* d_tri2trilist,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	int* d_caveshbdsize,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	trihandle cavesh, neighsh;
	REAL sign;

	int caveshbdsize = 0, j;

	cavesh = d_caveshlist[pos];
	for (j = 0; j < 3; j++)
	{
		if (!cudamesh_isshsubseg(cavesh, d_tri2seglist))
		{
			cudamesh_spivot(cavesh, neighsh, d_tri2trilist);
			if (neighsh.id != -1)
			{
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId + 1)
				{
					// A boundary edge
					sign = 1;
				}
				else
				{
					// Internal edge
					sign = -1;
				}
			}
			else
			{
				// A boundary edge
				sign = 1;
			}
		}
		else
		{
			// A segment. It is a boundary edge
			sign = 1;
		}
		if (sign >= 0)
		{
			caveshbdsize++;
		}
		cudamesh_senextself(cavesh);
	}

	d_caveshbdsize[pos] = caveshbdsize;
}

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
	int* d_caveshbdindices,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	trihandle cavesh, neighsh;
	REAL sign;

	int caveshbdsize = d_caveshbdsize[pos];
	if (caveshbdsize == 0)
		return;

	int prev = -1, newid = d_caveshbdindices[pos];
	int i = d_caveshhead[threadId], j;
	while (i != -1)
	{
		cavesh = d_caveshlist[i];
		for (j = 0; j < 3; j++)
		{
			if (!cudamesh_isshsubseg(cavesh, d_tri2seglist))
			{
				cudamesh_spivot(cavesh, neighsh, d_tri2trilist);
				if (neighsh.id != -1)
				{
					if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId)
					{
						// A boundary edge
						sign = 1;
					}
					else
					{
						// Internal edge
						sign = -1;
					}
				}
				else
				{
					// A boundary edge
					sign = 1;
				}
			}
			else
			{
				// A segment. It is a boundary edge
				sign = 1;
			}
			if (sign >= 0)
			{
				d_caveshbdlist[newid] = cavesh;
				d_caveshbdprev[newid] = prev;
				d_caveshbdnext[newid] = -1;
				if (prev != -1)
					d_caveshbdnext[prev] = newid;
				else
					d_caveshbdhead[threadId] = newid;
				prev = newid;
				newid++;
			}
			cudamesh_senextself(cavesh);
		}

		i = d_caveshnext[i];
		if (i == -1) // reach the end of list
		{
			d_caveshbdtail[threadId] = prev;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int caveshbdsize = d_caveshbdsize[pos];
	if (caveshbdsize == 0)
		return;

	trihandle cavesh, neighsh;
	REAL sign;

	int newid = d_caveshbdindices[pos];
	int j;
	
	cavesh = d_caveshlist[pos];
	for (j = 0; j < 3; j++)
	{
		if (!cudamesh_isshsubseg(cavesh, d_tri2seglist))
		{
			cudamesh_spivot(cavesh, neighsh, d_tri2trilist);
			if (neighsh.id != -1)
			{
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId + 1)
				{
					// A boundary edge
					sign = 1;
				}
				else
				{
					// Internal edge
					sign = -1;
				}
			}
			else
			{
				// A boundary edge
				sign = 1;
			}
		}
		else
		{
			// A segment. It is a boundary edge
			sign = 1;
		}
		if (sign >= 0)
		{
			d_caveshbdlist[newid] = cavesh;
			d_caveshbdidx[newid] = threadId;
			newid++;
		}
		cudamesh_senextself(cavesh);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int startidx = d_caveshbdindices[pos], newtriidx;
	int newptidx = oldpointsize + pos;
	trihandle parysh, checkseg, newsh, casin, casout, neighsh;
	int pa, pb;

	int i = d_caveshbdhead[threadId];
	while (i != -1)
	{
		parysh = d_caveshbdlist[i];
		cudamesh_sspivot(parysh, checkseg, d_tri2seglist);
		if ((parysh.shver & 01) != 0)
			cudamesh_sesymself(parysh);
		pa = cudamesh_sorg(parysh, d_trifacelist);
		pb = cudamesh_sdest(parysh, d_trifacelist);

		// Create a new subface
		newtriidx = d_emptytriindices[startidx++];
		newsh.id = newtriidx;
		newsh.shver = 0;
		cudamesh_setsorg(newsh, pa, d_trifacelist);
		cudamesh_setsdest(newsh, pb, d_trifacelist);
		cudamesh_setsapex(newsh, newptidx, d_trifacelist);
		d_tri2parentidxlist[newtriidx] = d_tri2parentidxlist[parysh.id];
		if (d_pointtypelist[pa] == FREEFACETVERTEX)
		{
			d_point2trilist[pa] = newsh;
		}
		if (d_pointtypelist[pb] == FREEFACETVERTEX)
		{
			d_point2trilist[pb] = newsh;
		}

		// Save the outer subfaces first
		cudamesh_spivot(parysh, casout, d_tri2trilist);
		d_casout[i] = casout;
		if (casout.id != -1)
		{
			casin = casout;
			if (checkseg.id != -1)
			{
				// Make sure that newsh has the right ori at this segment.
				checkseg.shver = 0;
				if (cudamesh_sorg(newsh, d_trifacelist) != cudamesh_sorg(checkseg, d_seglist))
				{
					cudamesh_sesymself(newsh);
					cudamesh_sesymself(parysh); // This side should also be inverse.
				}
				cudamesh_spivot(casin, neighsh, d_tri2trilist);
				while (neighsh.id != parysh.id)
				{
					casin = neighsh;
					cudamesh_spivot(casin, neighsh, d_tri2trilist);
				}
			}
			d_casin[i] = casin;
		}

		i = d_caveshbdnext[i];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshbdidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int newtriidx;
	int newptidx = oldpointsize + d_threadpos[threadId];
	trihandle parysh, checkseg, newsh, casin, casout, neighsh;
	int pa, pb;

	parysh = d_caveshbdlist[pos];
	cudamesh_sspivot(parysh, checkseg, d_tri2seglist);
	if ((parysh.shver & 01) != 0)
		cudamesh_sesymself(parysh);
	pa = cudamesh_sorg(parysh, d_trifacelist);
	pb = cudamesh_sdest(parysh, d_trifacelist);

	// Create a new subface
	newtriidx = d_emptytriindices[pos];
	newsh.id = newtriidx;
	newsh.shver = 0;
	cudamesh_setsorg(newsh, pa, d_trifacelist);
	cudamesh_setsdest(newsh, pb, d_trifacelist);
	cudamesh_setsapex(newsh, newptidx, d_trifacelist);
	d_tri2parentidxlist[newtriidx] = d_tri2parentidxlist[parysh.id];
	if (d_pointtypelist[pa] == FREEFACETVERTEX)
	{
		d_point2trilist[pa] = newsh;
	}
	if (d_pointtypelist[pb] == FREEFACETVERTEX)
	{
		d_point2trilist[pb] = newsh;
	}

	// Save the outer subfaces first
	cudamesh_spivot(parysh, casout, d_tri2trilist);
	d_casout[pos] = casout;
	if (casout.id != -1)
	{
		casin = casout;
		if (checkseg.id != -1)
		{
			// Make sure that newsh has the right ori at this segment.
			checkseg.shver = 0;
			if (cudamesh_sorg(newsh, d_trifacelist) != cudamesh_sorg(checkseg, d_seglist))
			{
				cudamesh_sesymself(newsh);
				cudamesh_sesymself(parysh); // This side should also be inverse.
			}
			cudamesh_spivot(casin, neighsh, d_tri2trilist);
			while (neighsh.id != parysh.id)
			{
				casin = neighsh;
				cudamesh_spivot(casin, neighsh, d_tri2trilist);
			}
		}
		d_casin[pos] = casin;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int startidx = d_caveshbdindices[pos], newtriidx;
	int newptidx = oldpointsize + pos;
	trihandle parysh, checkseg, newsh, casin, casout, neighsh;
	int pa, pb;

	int i = d_caveshbdhead[threadId];
	while (i != -1)
	{
		parysh = d_caveshbdlist[i];
		cudamesh_sspivot(parysh, checkseg, d_tri2seglist);
		if ((parysh.shver & 01) != 0)
			cudamesh_sesymself(parysh);

		// Create a new subface
		newtriidx = d_emptytriindices[startidx++];
		newsh.id = newtriidx;
		newsh.shver = 0;

		// Connect newsh to outer old subfaces (Phase 1).
		casout = d_casout[i];
		if (casout.id != -1)
		{
			//casin = casout;
			if (checkseg.id != -1)
			{
				// Make sure that newsh has the right ori at this segment.
				checkseg.shver = 0;
				if (cudamesh_sorg(newsh, d_trifacelist) != cudamesh_sorg(checkseg, d_seglist))
				{
					cudamesh_sesymself(newsh);
					cudamesh_sesymself(parysh); // This side should also be inverse.
				}
			}
			casin = d_casin[i];
			cudamesh_sbond1(newsh, casout, d_tri2trilist);
			cudamesh_sbond1(casin, newsh, d_tri2trilist);
		}

		i = d_caveshbdnext[i];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshbdidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int newtriidx;
	trihandle parysh, checkseg, newsh, casin, casout;
	int pa, pb;

	parysh = d_caveshbdlist[pos];
	cudamesh_sspivot(parysh, checkseg, d_tri2seglist);
	if ((parysh.shver & 01) != 0)
		cudamesh_sesymself(parysh);

	// Create a new subface
	newtriidx = d_emptytriindices[pos];
	newsh.id = newtriidx;
	newsh.shver = 0;

	// Connect newsh to outer old subfaces (Phase 1).
	casout = d_casout[pos];
	if (casout.id != -1)
	{
		//casin = casout;
		if (checkseg.id != -1)
		{
			// Make sure that newsh has the right ori at this segment.
			checkseg.shver = 0;
			if (cudamesh_sorg(newsh, d_trifacelist) != cudamesh_sorg(checkseg, d_seglist))
			{
				cudamesh_sesymself(newsh);
				cudamesh_sesymself(parysh); // This side should also be inverse.
			}
		}
		casin = d_casin[pos];
		cudamesh_sbond1(newsh, casout, d_tri2trilist);
		cudamesh_sbond1(casin, newsh, d_tri2trilist);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int startidx = d_caveshbdindices[pos], newtriidx;
	int newptidx = oldpointsize + pos;
	trihandle parysh, checkseg, newsh, casout;

	int i = d_caveshbdhead[threadId];
	while (i != -1)
	{
		parysh = d_caveshbdlist[i];
		cudamesh_sspivot(parysh, checkseg, d_tri2seglist);
		if ((parysh.shver & 01) != 0)
		{
			cudamesh_sesymself(parysh);
			d_caveshbdlist[i] = parysh; // Update the element in the list
		}

		// Create a new subface
		newtriidx = d_emptytriindices[startidx++];
		newsh.id = newtriidx;
		newsh.shver = 0;

		// Connect newsh to outer subfaces (Phase 2).
		// Check if old subface is connected to new one,
		// if so, fix it 
		cudamesh_spivot(parysh, casout, d_tri2trilist);

		if (casout.id != -1)
		{
			if (checkseg.id != -1)
			{
				// Make sure that newsh has the right ori at this segment.
				checkseg.shver = 0;
				if (cudamesh_sorg(newsh, d_trifacelist) != cudamesh_sorg(checkseg, d_seglist))
				{
					cudamesh_sesymself(newsh);
					cudamesh_sesymself(parysh); // This side should also be inverse.
					d_caveshbdlist[i] = parysh; // Update the element in the list
				}
			}
			if (d_tristatus[casout.id].isEmpty()) // old subface is connected to new one
			{
				cudamesh_sbond1(newsh, casout, d_tri2trilist);
			}
		}

		if (checkseg.id != -1)
		{
			cudamesh_ssbond(newsh, checkseg, d_tri2seglist, d_seg2trilist);
		}
		// Connect oldsh <== newsh (for connecting adjacent new subfaces).
		//   parysh and newsh point to the same edge and the same ori.
		cudamesh_sbond1(parysh, newsh, d_tri2trilist);

		i = d_caveshbdnext[i];
	}

	if (d_pointtypelist[newptidx] == FREEFACETVERTEX)
		d_point2trilist[newptidx] = newsh;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshbdidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int newtriidx;
	int newptidx = oldpointsize + d_threadpos[threadId];
	trihandle parysh, checkseg, newsh, casout;

	parysh = d_caveshbdlist[pos];
	cudamesh_sspivot(parysh, checkseg, d_tri2seglist);
	if ((parysh.shver & 01) != 0)
	{
		cudamesh_sesymself(parysh);
		d_caveshbdlist[pos] = parysh; // Update the element in the list
	}

	// Create a new subface
	newtriidx = d_emptytriindices[pos];
	newsh.id = newtriidx;
	newsh.shver = 0;

	// Connect newsh to outer subfaces (Phase 2).
	// Check if old subface is connected to new one,
	// if so, fix it 
	cudamesh_spivot(parysh, casout, d_tri2trilist);

	if (casout.id != -1)
	{
		if (checkseg.id != -1)
		{
			// Make sure that newsh has the right ori at this segment.
			checkseg.shver = 0;
			if (cudamesh_sorg(newsh, d_trifacelist) != cudamesh_sorg(checkseg, d_seglist))
			{
				cudamesh_sesymself(newsh);
				cudamesh_sesymself(parysh); // This side should also be inverse.
				d_caveshbdlist[pos] = parysh; // Update the element in the list
			}
		}
		if (d_tristatus[casout.id].isEmpty()) // old subface is connected to new one
		{
			cudamesh_sbond1(newsh, casout, d_tri2trilist);
		}
	}

	if (checkseg.id != -1)
	{
		cudamesh_ssbond(newsh, checkseg, d_tri2seglist, d_seg2trilist);
	}
	// Connect oldsh <== newsh (for connecting adjacent new subfaces).
	//   parysh and newsh point to the same edge and the same ori.
	cudamesh_sbond1(parysh, newsh, d_tri2trilist);

	if (d_pointtypelist[newptidx] == FREEFACETVERTEX)
		d_point2trilist[newptidx] = newsh;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	trihandle parysh, newsh, neighsh;
	int pa, pb;

	int i = d_caveshbdhead[threadId];
	while (i != -1)
	{
		// Get an old subface at edge [a, b].
		parysh = d_caveshbdlist[i];

		cudamesh_spivot(parysh, newsh, d_tri2trilist); // The new subface [a, b, p].
		cudamesh_senextself(newsh); // At edge [b, p].
		cudamesh_spivot(newsh, neighsh, d_tri2trilist);
		if (neighsh.id == -1) // No neighbor yet
		{
			// Find the adjacent new subface at edge [b, p].
			pb = cudamesh_sdest(parysh, d_trifacelist);
			neighsh = parysh;
			while (1) 
			{
				cudamesh_senextself(neighsh);
				cudamesh_spivotself(neighsh, d_tri2trilist);
				if (neighsh.id == -1) 
					break;
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId) 
					break;
				if (cudamesh_sdest(neighsh, d_trifacelist) != pb) 
					cudamesh_sesymself(neighsh);
			}
			if (neighsh.id != -1) 
			{
				// Now 'neighsh' is a new subface at edge [b, #].
				if (cudamesh_sorg(neighsh, d_trifacelist) != pb) 
					cudamesh_sesymself(neighsh);
				cudamesh_senext2self(neighsh); // Go to the open edge [p, b].
				cudamesh_sbond(newsh, neighsh, d_tri2trilist);
			}
			else {
				assert(false);
			}
		}

		cudamesh_spivot(parysh, newsh, d_tri2trilist); // The new subface [a, b, p].
		cudamesh_senext2self(newsh); // At edge [p, a].
		cudamesh_spivot(newsh, neighsh, d_tri2trilist);
		if (neighsh.id == -1) // No neighbor yet
		{
			// Find the adjacent new subface at edge [p, a].
			pa = cudamesh_sorg(parysh, d_trifacelist);
			neighsh = parysh;
			while (1)
			{
				cudamesh_senext2self(neighsh);
				cudamesh_spivotself(neighsh, d_tri2trilist);
				if (neighsh.id == -1)
					break;
				if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId)
					break;
				if (cudamesh_sorg(neighsh, d_trifacelist) != pa)
					cudamesh_sesymself(neighsh);
			}
			if (neighsh.id != -1)
			{
				// Now 'neighsh' is a new subface at edge [#, a].
				if (cudamesh_sdest(neighsh, d_trifacelist) != pa)
					cudamesh_sesymself(neighsh);
				cudamesh_senextself(neighsh); // Go to the open edge [a, p].
				cudamesh_sbond(newsh, neighsh, d_tri2trilist);
			}
			else {
				assert(false);
			}
		}

		// Update tristatus
		d_tristatus[parysh.id].clear();
		d_tristatus[newsh.id].setEmpty(false);

		i = d_caveshbdnext[i];
	}
}

__global__ void kernelConnectNewSubfaceNeighbors(
	int* d_trifacelist,
	trihandle* d_tri2trilist,
	tristatus* d_tristatus,
	trihandle* d_caveshbdlist,
	int* d_caveshbdidx,
	uint64* d_trimarker,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;
	
	int threadId = d_caveshbdidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	trihandle parysh, newsh, neighsh;
	int pa, pb;

	// Get an old subface at edge [a, b].
	parysh = d_caveshbdlist[pos];

	cudamesh_spivot(parysh, newsh, d_tri2trilist); // The new subface [a, b, p].
	cudamesh_senextself(newsh); // At edge [b, p].
	cudamesh_spivot(newsh, neighsh, d_tri2trilist);
	if (neighsh.id == -1) // No neighbor yet
	{
		// Find the adjacent new subface at edge [b, p].
		pb = cudamesh_sdest(parysh, d_trifacelist);
		neighsh = parysh;
		while (1)
		{
			cudamesh_senextself(neighsh);
			cudamesh_spivotself(neighsh, d_tri2trilist);
			if (neighsh.id == -1)
				break;
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId + 1)
				break;
			if (cudamesh_sdest(neighsh, d_trifacelist) != pb)
				cudamesh_sesymself(neighsh);
		}
		if (neighsh.id != -1)
		{
			// Now 'neighsh' is a new subface at edge [b, #].
			if (cudamesh_sorg(neighsh, d_trifacelist) != pb)
				cudamesh_sesymself(neighsh);
			cudamesh_senext2self(neighsh); // Go to the open edge [p, b].
			cudamesh_sbond(newsh, neighsh, d_tri2trilist);
		}
		else {
			assert(false);
		}
	}

	cudamesh_spivot(parysh, newsh, d_tri2trilist); // The new subface [a, b, p].
	cudamesh_senext2self(newsh); // At edge [p, a].
	cudamesh_spivot(newsh, neighsh, d_tri2trilist);
	if (neighsh.id == -1) // No neighbor yet
	{
		// Find the adjacent new subface at edge [p, a].
		pa = cudamesh_sorg(parysh, d_trifacelist);
		neighsh = parysh;
		while (1)
		{
			cudamesh_senext2self(neighsh);
			cudamesh_spivotself(neighsh, d_tri2trilist);
			if (neighsh.id == -1)
				break;
			if (cudamesh_getUInt64PriorityIndex(d_trimarker[neighsh.id]) != threadId + 1)
				break;
			if (cudamesh_sorg(neighsh, d_trifacelist) != pa)
				cudamesh_sesymself(neighsh);
		}
		if (neighsh.id != -1)
		{
			// Now 'neighsh' is a new subface at edge [#, a].
			if (cudamesh_sdest(neighsh, d_trifacelist) != pa)
				cudamesh_sesymself(neighsh);
			cudamesh_senextself(neighsh); // Go to the open edge [a, p].
			cudamesh_sbond(newsh, neighsh, d_tri2trilist);
		}
		else {
			assert(false);
		}
	}

	// Update tristatus
	d_tristatus[parysh.id].clear();
	d_tristatus[newsh.id].setEmpty(false);
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	trihandle parysh, cavesh, newsh, neighsh, casout;
	int newptidx = oldpointsize + pos;

	int i, j, head, next;
	i = head = d_cavesegshhead[threadId];
	bool onesubface = (head == d_cavesegshtail[threadId]);
	while (i != -1)
	{
		// Get the next old subface.
		next = d_cavesegshnext[i];
		// Get the saved old subface.
		parysh = d_cavesegshlist[i];
		// Get a possible new degenerated subface.
		cudamesh_spivot(parysh, cavesh, d_tri2trilist);
		if (cudamesh_sapex(cavesh, d_trifacelist) == newptidx) // a new degenerated subface
		{
			if (onesubface) // only one degenerated subface
			{
				for (j = 0; j < 2; j++)
				{
					cudamesh_senextself(cavesh);
					cudamesh_spivot(cavesh, newsh, d_tri2trilist);
					cudamesh_sdissolve(newsh, d_tri2trilist);
				}
			}
			else // more than one degenerated subface share at this segment
			{
				if (next == -1)
					parysh = d_cavesegshlist[head];
				else
					parysh = d_cavesegshlist[next];
				cudamesh_spivot(parysh, neighsh, d_tri2trilist);
				// Adjust cavesh and neighsh both at edge a->b, and has p as apex.
				if (cudamesh_sorg(neighsh, d_trifacelist) != cudamesh_sorg(cavesh, d_trifacelist))
				{
					cudamesh_sesymself(neighsh);
					assert(cudamesh_sorg(neighsh, d_trifacelist) == cudamesh_sorg(cavesh, d_trifacelist));
				}
				assert(cudamesh_sapex(neighsh, d_trifacelist) == newptidx);
				// Connect adjacent faces at two other edges of cavesh and neighsh.
				//   As a result, the two degenerated new faces are squeezed from the
				//   new triangulation of the cavity. Note that the squeezed faces
				//   still hold the adjacent informations which will be used in 
				//   re-connecting subsegments (if they exist).
				for (j = 0; j < 2; j++)
				{
					cudamesh_senextself(cavesh);
					cudamesh_senextself(neighsh);
					cudamesh_spivot(cavesh, newsh, d_tri2trilist);
					cudamesh_spivot(neighsh, casout, d_tri2trilist);
					cudamesh_sbond1(newsh, casout, d_tri2trilist);
				}
			}

			// Update tristatus
			d_tristatus[cavesh.id].clear(); // delete this degenerated subface

			// Update the point-to-subface map.
			if (d_pointtypelist[newptidx] == FREEFACETVERTEX)
				d_point2trilist[newptidx] = newsh;
		}

		i = next;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavesegshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	trihandle parysh, cavesh, newsh, neighsh, casout;
	int newptidx = oldpointsize + d_threadpos[threadId];

	int j, head, next, size;
	size = d_initialsubcavitysize[threadId];
	bool onesubface = (size == 1);

	// Get the next old subface.
	next = pos + 1;
	if (next >= numofthreads)
		next = pos - size + 1;
	else
	{
		int nextidx = d_cavesegshidx[next];
		if (nextidx != threadId)
			next = pos - size + 1;
	}
	// Get the saved old subface.
	parysh = d_cavesegshlist[pos];
	// Get a possible new degenerated subface.
	cudamesh_spivot(parysh, cavesh, d_tri2trilist);
	if (cudamesh_sapex(cavesh, d_trifacelist) == newptidx) // a new degenerated subface
	{
		if (onesubface) // only one degenerated subface
		{
			for (j = 0; j < 2; j++)
			{
				cudamesh_senextself(cavesh);
				cudamesh_spivot(cavesh, newsh, d_tri2trilist);
				cudamesh_sdissolve(newsh, d_tri2trilist);
			}
		}
		else // more than one degenerated subface share at this segment
		{
			parysh = d_cavesegshlist[next];
			cudamesh_spivot(parysh, neighsh, d_tri2trilist);
			// Adjust cavesh and neighsh both at edge a->b, and has p as apex.
			if (cudamesh_sorg(neighsh, d_trifacelist) != cudamesh_sorg(cavesh, d_trifacelist))
			{
				cudamesh_sesymself(neighsh);
				assert(cudamesh_sorg(neighsh, d_trifacelist) == cudamesh_sorg(cavesh, d_trifacelist));
			}
			assert(cudamesh_sapex(neighsh, d_trifacelist) == newptidx);
			// Connect adjacent faces at two other edges of cavesh and neighsh.
			//   As a result, the two degenerated new faces are squeezed from the
			//   new triangulation of the cavity. Note that the squeezed faces
			//   still hold the adjacent informations which will be used in 
			//   re-connecting subsegments (if they exist).
			for (j = 0; j < 2; j++)
			{
				cudamesh_senextself(cavesh);
				cudamesh_senextself(neighsh);
				cudamesh_spivot(cavesh, newsh, d_tri2trilist);
				cudamesh_spivot(neighsh, casout, d_tri2trilist);
				cudamesh_sbond1(newsh, casout, d_tri2trilist);
			}
		}

		// Update tristatus
		d_tristatus[cavesh.id].clear(); // delete this degenerated subface

		// Update the point-to-subface map.
		if (d_pointtypelist[newptidx] == FREEFACETVERTEX)
			d_point2trilist[newptidx] = newsh;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	int newptidx = oldpointsize + pos;
	trihandle splitseg, aseg, bseg, aoutseg, boutseg;
	int pa, pb;

	splitseg = trihandle(d_segidlist[threadId], 0);
	pa = cudamesh_sorg(splitseg, d_seglist);
	pb = cudamesh_sdest(splitseg, d_seglist);

	// Set new segments
	aseg.id = d_emptysegindices[2 * pos];
	aseg.shver = 0;
	bseg.id = d_emptysegindices[2 * pos + 1];
	bseg.shver = 0;

	cudamesh_setsorg(aseg, pa, d_seglist);
	cudamesh_setsdest(aseg, newptidx, d_seglist);
	cudamesh_setsapex(aseg, -1, d_seglist);
	cudamesh_setsorg(bseg, newptidx, d_seglist);
	cudamesh_setsdest(bseg, pb, d_seglist);
	cudamesh_setsapex(bseg, -1, d_seglist);

	d_seg2parentidxlist[aseg.id] = d_seg2parentidxlist[splitseg.id];
	d_seg2parentidxlist[bseg.id] = d_seg2parentidxlist[splitseg.id];

	// Update segstatus
	d_segstatus[splitseg.id].clear();
	d_segstatus[aseg.id].setEmpty(false);
	d_segstatus[bseg.id].setEmpty(false);

	// Reset segment encroachement marker
	d_segencmarker[splitseg.id] = -1;

	// Connect [#, a]<->[a, p]. It is possible that [#, a] is an old segment to be removed
	cudamesh_senext2(splitseg, boutseg); // Temporarily use boutseg.
	cudamesh_spivotself(boutseg, d_seg2trilist);
	if (boutseg.id != -1) 
	{
		cudamesh_senext2(aseg, aoutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Connect [p, b]<->[b, #]. It is possible that [b, #] is an old segment to be removed
	cudamesh_senext(splitseg, aoutseg);
	cudamesh_spivotself(aoutseg, d_seg2trilist);
	if (aoutseg.id != -1) 
	{
		cudamesh_senext(bseg, boutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Connect [a, p] <-> [p, b].
	cudamesh_senext(aseg, aoutseg);
	cudamesh_senext2(bseg, boutseg);
	cudamesh_sbond(aoutseg, boutseg, d_seg2trilist);

	// Connect subsegs [a, p] and [p, b] to adjacent new subfaces.
	// Although the degenerated new faces have been squeezed. They still
	//   hold the connections to the actual new faces.
	trihandle parysh, neighsh, newsh;
	int i = d_cavesegshhead[threadId];
	while (i != -1)
	{
		parysh = d_cavesegshlist[i];
		cudamesh_spivot(parysh, neighsh, d_tri2trilist);
		// neighsh is a degenerated new face.
		if (cudamesh_sorg(neighsh, d_trifacelist) != pa) 
		{
			cudamesh_sesymself(neighsh);
		}
		cudamesh_senext2(neighsh, newsh);
		cudamesh_spivotself(newsh, d_tri2trilist); // The edge [p, a] in newsh
		cudamesh_ssbond(newsh, aseg, d_tri2seglist, d_seg2trilist);
		cudamesh_senext(neighsh, newsh);
		cudamesh_spivotself(newsh, d_tri2trilist); // The edge [b, p] in newsh
		cudamesh_ssbond(newsh, bseg, d_tri2seglist, d_seg2trilist);

		i = d_cavesegshnext[i];
	}

	if (d_pointtypelist[newptidx] == FREESEGVERTEX)
		d_point2trilist[newptidx] = aseg;
	if (d_pointtypelist[pa] == FREESEGVERTEX)
		d_point2trilist[pa] = aseg;
	if (d_pointtypelist[pb] == FREESEGVERTEX)
		d_point2trilist[pb] = bseg;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	int newptidx = oldpointsize + pos;
	trihandle splitseg, aseg, bseg, aoutseg, boutseg;
	int pa, pb;

	splitseg = trihandle(d_segidlist[threadId], 0);
	pa = cudamesh_sorg(splitseg, d_seglist);
	pb = cudamesh_sdest(splitseg, d_seglist);

	// Set new segments
	aseg.id = d_emptysegindices[2 * pos];
	aseg.shver = 0;
	bseg.id = d_emptysegindices[2 * pos + 1];
	bseg.shver = 0;

	cudamesh_setsorg(aseg, pa, d_seglist);
	cudamesh_setsdest(aseg, newptidx, d_seglist);
	cudamesh_setsapex(aseg, -1, d_seglist);
	cudamesh_setsorg(bseg, newptidx, d_seglist);
	cudamesh_setsdest(bseg, pb, d_seglist);
	cudamesh_setsapex(bseg, -1, d_seglist);

	d_seg2parentidxlist[aseg.id] = d_seg2parentidxlist[splitseg.id];
	d_seg2parentidxlist[bseg.id] = d_seg2parentidxlist[splitseg.id];

	// Update segstatus
	d_segstatus[splitseg.id].clear();
	d_segstatus[aseg.id].setEmpty(false);
	d_segstatus[bseg.id].setEmpty(false);

	// Reset segment encroachement marker
	d_segencmarker[splitseg.id] = -1;

	// Connect [#, a]<->[a, p]. It is possible that [#, a] is an old segment to be removed
	cudamesh_senext2(splitseg, boutseg); // Temporarily use boutseg.
	cudamesh_spivotself(boutseg, d_seg2trilist);
	if (boutseg.id != -1)
	{
		cudamesh_senext2(aseg, aoutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Connect [p, b]<->[b, #]. It is possible that [b, #] is an old segment to be removed
	cudamesh_senext(splitseg, aoutseg);
	cudamesh_spivotself(aoutseg, d_seg2trilist);
	if (aoutseg.id != -1)
	{
		cudamesh_senext(bseg, boutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Connect [a, p] <-> [p, b].
	cudamesh_senext(aseg, aoutseg);
	cudamesh_senext2(bseg, boutseg);
	cudamesh_sbond(aoutseg, boutseg, d_seg2trilist);

	if (d_pointtypelist[newptidx] == FREESEGVERTEX)
		d_point2trilist[newptidx] = aseg;
	if (d_pointtypelist[pa] == FREESEGVERTEX)
		d_point2trilist[pa] = aseg;
	if (d_pointtypelist[pb] == FREESEGVERTEX)
		d_point2trilist[pb] = bseg;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavesegshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	int tpos = d_threadpos[threadId];
	int newptidx = oldpointsize + tpos;
	trihandle splitseg, aseg, bseg;
	int pa, pb;

	splitseg = trihandle(d_segidlist[threadId], 0);
	pa = cudamesh_sorg(splitseg, d_seglist);
	pb = cudamesh_sdest(splitseg, d_seglist);

	// Set new segments
	aseg.id = d_emptysegindices[2 * tpos];
	aseg.shver = 0;
	bseg.id = d_emptysegindices[2 * tpos + 1];
	bseg.shver = 0;

	// Connect subsegs [a, p] and [p, b] to adjacent new subfaces.
	// Although the degenerated new faces have been squeezed. They still
	//   hold the connections to the actual new faces.
	trihandle parysh, neighsh, newsh;
	parysh = d_cavesegshlist[pos];
	cudamesh_spivot(parysh, neighsh, d_tri2trilist);
	// neighsh is a degenerated new face.
	if (cudamesh_sorg(neighsh, d_trifacelist) != pa)
	{
		cudamesh_sesymself(neighsh);
	}
	cudamesh_senext2(neighsh, newsh);
	cudamesh_spivotself(newsh, d_tri2trilist); // The edge [p, a] in newsh
	cudamesh_ssbond(newsh, aseg, d_tri2seglist, d_seg2trilist);
	cudamesh_senext(neighsh, newsh);
	cudamesh_spivotself(newsh, d_tri2trilist); // The edge [b, p] in newsh
	cudamesh_ssbond(newsh, bseg, d_tri2seglist, d_seg2trilist);
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	trihandle splitseg, aseg, bseg, aoutseg, boutseg;

	// Get old and new segments
	splitseg = trihandle(d_segidlist[threadId], 0);
	aseg.id = d_emptysegindices[2 * pos];
	aseg.shver = 0;
	bseg.id = d_emptysegindices[2 * pos + 1];
	bseg.shver = 0;

	// Connect [#, a]<->[a, p]. 
	// If [a, b] is connected to a new segment [#, a], 
	// then it is possible that [a, p] is connected to an old segment [*, a].
	// Fix it.
	cudamesh_senext2(splitseg, boutseg);
	cudamesh_spivotself(boutseg, d_seg2trilist);
	if (boutseg.id != -1 && d_segmarker[boutseg.id] == MAXINT)
	{
		cudamesh_senext2(aseg, aoutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Connect [p, b]<->[b, #].
	// if [a, b] is connected to a new segment [b, #],
	// then it is possible that [p, b] is connected to an old segment [b, *].
	// Fix it.
	cudamesh_senext(splitseg, aoutseg);
	cudamesh_spivotself(aoutseg, d_seg2trilist);
	if (aoutseg.id != -1 && d_segmarker[aoutseg.id] == MAXINT)
	{
		cudamesh_senext(bseg, boutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Add new segments into list
	int newidx = 2 * pos;
	d_cavesegshhead[threadId] = newidx;
	d_cavesegshtail[threadId] = newidx + 1;
	d_cavesegshlist[newidx] = aseg;
	d_cavesegshprev[newidx] = -1;
	d_cavesegshnext[newidx] = newidx + 1;
	d_cavesegshlist[newidx + 1] = bseg;
	d_cavesegshprev[newidx + 1] = newidx;
	d_cavesegshnext[newidx + 1] = -1;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	trihandle splitseg, aseg, bseg, aoutseg, boutseg;

	// Get old and new segments
	splitseg = trihandle(d_segidlist[threadId], 0);
	aseg.id = d_emptysegindices[2 * pos];
	aseg.shver = 0;
	bseg.id = d_emptysegindices[2 * pos + 1];
	bseg.shver = 0;

	// Connect [#, a]<->[a, p]. 
	// If [a, b] is connected to a new segment [#, a], 
	// then it is possible that [a, p] is connected to an old segment [*, a].
	// Fix it.
	cudamesh_senext2(splitseg, boutseg);
	cudamesh_spivotself(boutseg, d_seg2trilist);
	if (boutseg.id != -1 && d_segmarker[boutseg.id] == 0)
	{
		cudamesh_senext2(aseg, aoutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Connect [p, b]<->[b, #].
	// if [a, b] is connected to a new segment [b, #],
	// then it is possible that [p, b] is connected to an old segment [b, *].
	// Fix it.
	cudamesh_senext(splitseg, aoutseg);
	cudamesh_spivotself(aoutseg, d_seg2trilist);
	if (aoutseg.id != -1 && d_segmarker[aoutseg.id] == 0)
	{
		cudamesh_senext(bseg, boutseg);
		cudamesh_sbond(boutseg, aoutseg, d_seg2trilist);
	}

	// Add new segments into list
	int newidx = 2 * pos;
	d_cavesegshlist[newidx] = aseg;
	d_cavesegshidx[newidx] = threadId;
	d_cavesegshlist[newidx + 1] = bseg;
	d_cavesegshidx[newidx + 1] = threadId;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int newptidx = oldpointsize + pos;
	trihandle parysh, checksh;
	tethandle neightet, spintet;

	int i = d_caveshbdhead[threadId], j;
	while (i != -1)
	{
		// Get an old subface at edge [a, b].
		parysh = d_caveshbdlist[i];
		cudamesh_spivot(parysh, checksh, d_tri2trilist); // The new subface [a, b, p].
	    // Do not recover a deleted new face (degenerated).
		if (!d_tristatus[checksh.id].isEmpty()) 
		{
			// Note that the old subface still connects to adjacent old tets 
			//   of C(p), which still connect to the tets outside C(p).
			cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
			//assert(d_tetmarker[neightet.id] == threadId);
			// Find the adjacent tet containing the edge [a,b] outside C(p).
			spintet = neightet;
			while (1) 
			{
				cudamesh_fnextself(spintet, d_neighborlist);
				//printf("spintet %d\n", spintet.id);
				if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId) 
					break;
				assert(spintet.id != neightet.id);
			}
			// The adjacent tet connects to a new tet in C(p).
			cudamesh_fsym(spintet, neightet, d_neighborlist);
			//assert(d_tetmarker[neightet.id] != threadId);
			// Find the tet containing the face [a, b, p].
			spintet = neightet;
			while (1) 
			{
				cudamesh_fnextself(spintet, d_neighborlist);
				if (cudamesh_apex(spintet, d_tetlist) == newptidx) 
					break;
				assert(spintet.id != neightet.id);
			}
			// Adjust the edge direction in spintet and checksh.
			if (cudamesh_sorg(checksh, d_trifacelist) != cudamesh_org(spintet, d_tetlist)) 
			{
				cudamesh_sesymself(checksh);
				assert(cudamesh_sorg(checksh, d_trifacelist) == cudamesh_org(spintet, d_tetlist));
			}
			assert(cudamesh_sdest(checksh, d_trifacelist) == cudamesh_dest(spintet, d_tetlist));
			// Connect the subface to two adjacent tets.
			cudamesh_tsbond(spintet, checksh, d_tet2trilist, d_tri2tetlist);
			cudamesh_fsymself(spintet, d_neighborlist);
			cudamesh_sesymself(checksh);
			cudamesh_tsbond(spintet, checksh, d_tet2trilist, d_tri2tetlist);
		}
		else
		{
			// A deleted degenerated subface
			// Clear all neighbor information
			for (j = 0; j < 2; j++)
			{
				d_tri2tetlist[2 * checksh.id + j] = tethandle(-1, 11);
			}

			for (j = 0; j < 3; j++)
			{
				d_tri2trilist[3 * checksh.id + j] = trihandle(-1, 0);
			}

			for (j = 0; j < 3; j++)
			{
				d_tri2seglist[3 * checksh.id + j] = trihandle(-1, 0);
			}
		}

		i = d_caveshbdnext[i];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshbdidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 2)
		return;

	int newptidx = oldpointsize + d_threadpos[threadId];
	trihandle parysh, checksh;
	tethandle neightet, spintet;

	int j;
	// Get an old subface at edge [a, b].
	parysh = d_caveshbdlist[pos];
	cudamesh_spivot(parysh, checksh, d_tri2trilist); // The new subface [a, b, p].
	// Do not recover a deleted new face (degenerated).
	if (!d_tristatus[checksh.id].isEmpty())
	{
		// Note that the old subface still connects to adjacent old tets 
		//   of C(p), which still connect to the tets outside C(p).
		cudamesh_stpivot(parysh, neightet, d_tri2tetlist);
		//assert(d_tetmarker[neightet.id] == threadId);
		// Find the adjacent tet containing the edge [a,b] outside C(p).
		spintet = neightet;
		while (1)
		{
			cudamesh_fnextself(spintet, d_neighborlist);
			//printf("spintet %d\n", spintet.id);
			if (cudamesh_getUInt64PriorityIndex(d_tetmarker[spintet.id]) != threadId + 1)
				break;
			assert(spintet.id != neightet.id);
		}
		// The adjacent tet connects to a new tet in C(p).
		cudamesh_fsym(spintet, neightet, d_neighborlist);
		//assert(d_tetmarker[neightet.id] != threadId);
		// Find the tet containing the face [a, b, p].
		spintet = neightet;
		while (1)
		{
			cudamesh_fnextself(spintet, d_neighborlist);
			if (cudamesh_apex(spintet, d_tetlist) == newptidx)
				break;
			assert(spintet.id != neightet.id);
		}
		// Adjust the edge direction in spintet and checksh.
		if (cudamesh_sorg(checksh, d_trifacelist) != cudamesh_org(spintet, d_tetlist))
		{
			cudamesh_sesymself(checksh);
			assert(cudamesh_sorg(checksh, d_trifacelist) == cudamesh_org(spintet, d_tetlist));
		}
		assert(cudamesh_sdest(checksh, d_trifacelist) == cudamesh_dest(spintet, d_tetlist));
		// Connect the subface to two adjacent tets.
		cudamesh_tsbond(spintet, checksh, d_tet2trilist, d_tri2tetlist);
		cudamesh_fsymself(spintet, d_neighborlist);
		cudamesh_sesymself(checksh);
		cudamesh_tsbond(spintet, checksh, d_tet2trilist, d_tri2tetlist);
	}
	else
	{
		// A deleted degenerated subface
		// Clear all neighbor information
		for (j = 0; j < 2; j++)
		{
			d_tri2tetlist[2 * checksh.id + j] = tethandle(-1, 11);
		}

		for (j = 0; j < 3; j++)
		{
			d_tri2trilist[3 * checksh.id + j] = trihandle(-1, 0);
		}

		for (j = 0; j < 3; j++)
		{
			d_tri2seglist[3 * checksh.id + j] = trihandle(-1, 0);
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	trihandle checkseg, checksh;
	tethandle neightet, spintet;

	int i = d_cavesegshhead[threadId];
	while (i != -1)
	{
		checkseg = d_cavesegshlist[i];
		// Get the adjacent new subface.
		checkseg.shver = 0;
		cudamesh_spivot(checkseg, checksh, d_seg2trilist);;
		if (checksh.id != -1) 
		{
			// Get the adjacent new tetrahedron.
			cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
		}
		else
		{
			// It's a dangling segment.
			cudamesh_point2tetorg(cudamesh_sorg(checkseg, d_seglist), neightet, d_point2tetlist, d_tetlist);
			cudamesh_finddirection(&neightet, cudamesh_sdest(checkseg, d_seglist), d_pointlist, d_tetlist, d_neighborlist, d_randomseed + pos);
			assert(cudamesh_dest(neightet, d_tetlist) == cudamesh_sdest(checkseg, d_seglist));
		}
		//assert(d_tetmarker[neightet.id] != threadId);
		cudamesh_sstbond1(checkseg, neightet, d_seg2tetlist);
		spintet = neightet;
		while (1) 
		{
			cudamesh_tssbond1(spintet, checkseg, d_tet2seglist);
			cudamesh_fnextself(spintet, d_neighborlist);
			if (spintet.id == neightet.id) break;
		}

		i = d_cavesegshnext[i];
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavesegshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	unsigned long randomseed = 1;
	trihandle checkseg, checksh;
	tethandle neightet, spintet;

	checkseg = d_cavesegshlist[pos];
	// Get the adjacent new subface.
	checkseg.shver = 0;
	cudamesh_spivot(checkseg, checksh, d_seg2trilist);;
	if (checksh.id != -1)
	{
		// Get the adjacent new tetrahedron.
		cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
	}
	else
	{
		// It's a dangling segment.
		cudamesh_point2tetorg(cudamesh_sorg(checkseg, d_seglist), neightet, d_point2tetlist, d_tetlist);
		cudamesh_finddirection(&neightet, cudamesh_sdest(checkseg, d_seglist), d_pointlist, d_tetlist, d_neighborlist, &randomseed);
		assert(cudamesh_dest(neightet, d_tetlist) == cudamesh_sdest(checkseg, d_seglist));
	}
	//assert(d_tetmarker[neightet.id] != threadId);
	cudamesh_sstbond1(checkseg, neightet, d_seg2tetlist);
	spintet = neightet;
	while (1)
	{
		cudamesh_tssbond1(spintet, checkseg, d_tet2seglist);
		cudamesh_fnextself(spintet, d_neighborlist);
		if (spintet.id == neightet.id) break;
	}
}


__global__ void kernelResetOldSubsegInfo(
	int* d_segidlist,
	int* d_threadlist,
	int* d_threadmarker,
	trihandle* d_seg2trilist,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	int segid = d_segidlist[threadId];
	for (int j = 0; j < 3; j++)
	{
		d_seg2trilist[3 * segid + j] = trihandle(-1, 0);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos], j;
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0 && threadmarker != 1)
		return;

	trihandle checksh;

	int i = d_caveshhead[threadId];
	while (i != -1)
	{
		checksh = d_caveshlist[i];
		d_tristatus[checksh.id].clear();
		d_subfaceencmarker[checksh.id] = -1;

		for (j = 0; j < 3; j++)
		{
			d_tri2trilist[3 * checksh.id + j] = trihandle(-1, 0); // reset neighbor to empty
		}

		for (j = 0; j < 3; j++)
		{
			d_tri2seglist[3 * checksh.id + j] = trihandle(-1, 0);
		}

		i = d_caveshnext[i];
	}
}

__global__ void kernelResetOldSubfaceInfo(
	trihandle* d_tri2trilist,
	trihandle* d_tri2seglist,
	tristatus* d_tristatus,
	int* d_subfaceencmarker,
	trihandle* d_caveshlist,
	int* d_caveshidx,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshidx[pos], j;
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0 && threadmarker != 1)
		return;

	trihandle checksh;
	checksh = d_caveshlist[pos];
	d_tristatus[checksh.id].clear();
	d_subfaceencmarker[checksh.id] = -1;

	for (j = 0; j < 3; j++)
	{
		d_tri2trilist[3 * checksh.id + j] = trihandle(-1, 0); // reset neighbor to empty
	}

	for (j = 0; j < 3; j++)
	{
		d_tri2seglist[3 * checksh.id + j] = trihandle(-1, 0);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos], j;
	tethandle checktet;

	int i = d_caveoldtethead[threadId];
	while (i != -1)
	{
		checktet = d_caveoldtetlist[i];
		d_tetstatus[checktet.id].clear();
		for (j = 0; j < 4; j++)
		{
			d_neighborlist[4 * checktet.id + j] = tethandle(-1, 11); // reset neighbor to empty
		}
		for (j = 0; j < 4; j++)
		{
			d_tet2trilist[4 * checktet.id + j] = trihandle(-1, 0); // reset subface to empty
		}
		for (j = 0; j < 6; j++)
		{
			d_tet2seglist[6 * checktet.id + j] = trihandle(-1, 0); // reset subseg to empty
		}

		i = d_caveoldtetnext[i];
	}
}

__global__ void kernelResetOldTetInfo(
	tethandle* d_neighborlist,
	trihandle* d_tet2trilist,
	trihandle* d_tet2seglist,
	tetstatus* d_tetstatus,
	tethandle* d_caveoldtetlist,
	int* d_caveoldtetidx,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveoldtetidx[pos], j;
	tethandle checktet;

	checktet = d_caveoldtetlist[pos];
	d_tetstatus[checktet.id].clear();
	for (j = 0; j < 4; j++)
	{
		d_neighborlist[4 * checktet.id + j] = tethandle(-1, 11); // reset neighbor to empty
	}
	for (j = 0; j < 4; j++)
	{
		d_tet2trilist[4 * checktet.id + j] = trihandle(-1, 0); // reset subface to empty
	}
	for (j = 0; j < 6; j++)
	{
		d_tet2seglist[6 * checktet.id + j] = trihandle(-1, 0); // reset subseg to empty
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	trihandle checkseg;
	int i, encpt;

	// Check all segments outside cavity
	i = d_cavetetseghead[threadId];
	while (i != -1)
	{
		checkseg = d_cavetetseglist[i];

		if (d_segmarker[checkseg.id] != threadId) // Not a splitting segment
		{
			checkseg4split(
				&checkseg, encpt,
				d_pointlist, d_seglist, d_seg2tetlist, d_tetlist, d_neighborlist);
			d_segencmarker[checkseg.id] = encpt;
		}

		i = d_cavetetsegnext[i];
	}

	// Check new segments when it is segment point insertion.
	// In this case, new segments are stored in cavesegshlist
	if (threadmarker == 0)
	{
		i = d_cavesegshhead[threadId];
		while (i != -1)
		{
			checkseg = d_cavesegshlist[i];

			checkseg4split(
				&checkseg, encpt,
				d_pointlist, d_seglist, d_seg2tetlist, d_tetlist, d_neighborlist);
			d_segencmarker[checkseg.id] = encpt;

			i = d_cavesegshnext[i];
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetsegidx[pos];
	int threadmarker = d_threadmarker[threadId];

	trihandle checkseg;
	int encpt;

	// Check all segments outside cavity
	checkseg = d_cavetetseglist[pos];

	if (d_segmarker[checkseg.id] != threadId + 1) // Not a splitting segment
	{
		checkseg4split(
			&checkseg, encpt,
			d_pointlist, d_seglist, d_seg2tetlist, d_tetlist, d_neighborlist);
		d_segencmarker[checkseg.id] = encpt;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavesegshidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0)
		return;

	trihandle checkseg;
	int encpt;

	// Check new segments when it is segment point insertion.
	// In this case, new segments are stored in cavesegshlist
	checkseg = d_cavesegshlist[pos];
	checkseg4split(
		&checkseg, encpt,
		d_pointlist, d_seglist, d_seg2tetlist, d_tetlist, d_neighborlist);
	d_segencmarker[checkseg.id] = encpt;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];

	trihandle checkfac;
	int i, encpt;

	// Check all subfaces outside cavity
	i = d_cavetetshhead[threadId];
	while (i != -1)
	{
		checkfac = d_cavetetshlist[i];

		if (cudamesh_getUInt64PriorityIndex(d_trimarker[checkfac.id]) != threadId) // Not a splitting subface
		{
			checkface4split(
				&checkfac, encpt,
				d_pointlist, d_trifacelist, d_tri2tetlist, d_tetlist);
			d_subfaceencmarker[checkfac.id] = encpt;
		}

		i = d_cavetetshnext[i];
	}

	// Check new subfaces when it is segment/subface point insertion.
	// In this case, new subfaces are connected to old subfaces in caveshbdlist
	if (threadmarker == 0 || threadmarker == 1)
	{
		trihandle parysh;
		i = d_caveshbdhead[threadId];
		while (i != -1)
		{
			parysh = d_caveshbdlist[i];
			cudamesh_spivot(parysh, checkfac, d_tri2trilist);

			if (!d_tristatus[checkfac.id].isEmpty())
			{
				checkface4split(
					&checkfac, encpt,
					d_pointlist, d_trifacelist, d_tri2tetlist, d_tetlist);
				d_subfaceencmarker[checkfac.id] = encpt;
			}

			i = d_caveshbdnext[i];
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavetetshidx[pos];
	int threadmarker = d_threadmarker[threadId];

	trihandle checkfac;
	int encpt;

	// Check all subfaces outside cavity
	checkfac = d_cavetetshlist[pos];

	if (cudamesh_getUInt64PriorityIndex(d_trimarker[checkfac.id]) != threadId + 1) // Not a splitting subface
	{
		checkface4split(
			&checkfac, encpt,
			d_pointlist, d_trifacelist, d_tri2tetlist, d_tetlist);
		d_subfaceencmarker[checkfac.id] = encpt;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_caveshbdidx[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker != 0 && threadmarker != 1)
		return;

	trihandle checkfac;
	int encpt;

	// Check new subfaces when it is segment/subface point insertion.
	// In this case, new subfaces are connected to old subfaces in caveshbdlist
	trihandle parysh = d_caveshbdlist[pos];
	cudamesh_spivot(parysh, checkfac, d_tri2trilist);
	if (!d_tristatus[checkfac.id].isEmpty())
	{
		checkface4split(
			&checkfac, encpt,
			d_pointlist, d_trifacelist, d_tri2tetlist, d_tetlist);
		d_subfaceencmarker[checkfac.id] = encpt;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	tethandle cavetet;

	int i = d_cavebdryhead[threadId];
	while (i != -1)
	{
		cavetet = d_cavebdrylist[i];
		if (cavetet.id != -1) // cavetet.id may be -1 because of redundency
		{
			if (checktet4split(&cavetet, d_pointlist, d_tetlist, minratio))
				d_tetstatus[cavetet.id].setBad(true);
		}
		i = d_cavebdrynext[i];
	}
}

__global__ void kernelUpdateTetBadstatus(
	REAL* d_pointlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	tethandle* d_cavebdrylist,
	int* d_cavebdryidx,
	REAL minratio,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_cavebdryidx[pos];
	tethandle cavetet;
	cavetet = d_cavebdrylist[pos];
	if (cavetet.id != -1) // cavetet.id may be -1 because of redundency
	{
		if (checktet4split(&cavetet, d_pointlist, d_tetlist, minratio))
			d_tetstatus[cavetet.id].setBad(true);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	if (threadmarker == 0)
	{
		int newptidx = oldpointsize + pos;
		int parentptidx = d_parentpt[threadId];

		REAL rv = d_smlen[threadId], rp;
		verttype parenttype = d_pointtypelist[parentptidx];

		if (parenttype == FREESEGVERTEX)
		{
			trihandle parentseg1, parentseg2;
			parentseg1 = d_point2trilist[newptidx];
			parentseg2 = d_point2trilist[parentptidx];

			if (cudamesh_segsegadjacent(parentseg1.id, parentseg2.id, d_seg2parentidxlist, d_segparentendpointidxlist))
			{
				rp = d_pointradius[parentptidx];
				if (rv < rp)
					rv = rp; // The relaxed insertion radius of new point
			}
		}
		else if (parenttype == FREEFACETVERTEX)
		{
			trihandle parentseg, parentsh;
			parentseg = d_point2trilist[newptidx];
			parentsh = d_point2trilist[parentptidx];
			if (cudamesh_segfacetadjacent(parentseg.id, parentsh.id, d_seg2parentidxlist, d_segparentendpointidxlist,
				d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
			{
				rp = d_pointradius[parentptidx];
				if (rv < rp)
					rv = rp; // The relaxed insertion radius of new point
			}

		}
		d_pointradius[newptidx] = rv;
	}
	else if (threadmarker == 1)
	{
		int newptidx = oldpointsize + pos;
		int parentptidx = d_parentpt[threadId];

		REAL rv = d_smlen[threadId], rp;
		verttype parenttype = d_pointtypelist[parentptidx];

		if (parenttype == FREESEGVERTEX)
		{
			trihandle parentseg, parentsh;
			parentseg = d_point2trilist[parentptidx];
			parentsh = d_point2trilist[newptidx];

			if (cudamesh_segfacetadjacent(parentseg.id, parentsh.id, d_seg2parentidxlist, d_segparentendpointidxlist,
				d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
			{
				rp = d_pointradius[parentptidx];
				if (rv < (sqrt(2.0) * rp))
					rv = sqrt(2.0) * rp; // The relaxed insertion radius of new point
			}
		}
		else if (parenttype == FREEFACETVERTEX)
		{
			trihandle parentsh1, parentsh2;
			parentsh1 = d_point2trilist[parentptidx];
			parentsh2 = d_point2trilist[newptidx];
			if (cudamesh_facetfacetadjacent(parentsh1.id, parentsh2.id,
				d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
			{
				rp = d_pointradius[parentptidx];
				if (rv < rp)
					rv = rp; // The relaxed insertion radius of new point
			}

		}
		d_pointradius[newptidx] = rv;
	}
	else
	{
		int splittetid = d_insertidxlist[threadId];
		tethandle splittet(splittetid, 11);

		int newptidx = oldpointsize + pos;
		REAL *newpt = cudamesh_id2pointlist(newptidx, d_pointlist);

		int orgidx = cudamesh_org(splittet, d_tetlist);
		REAL *org = cudamesh_id2pointlist(orgidx, d_pointlist);

		REAL rv = cudamesh_distance(newpt, org);
		d_pointradius[newptidx] = rv;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int newptidx = oldpointsize + pos;
	int parentptidx = d_parentpt[threadId];

	REAL rv = d_smlen[threadId], rp;
	verttype parenttype = d_pointtypelist[parentptidx];

	if (parenttype == FREESEGVERTEX)
	{
		trihandle parentseg1, parentseg2;
		parentseg1 = d_point2trilist[newptidx];
		parentseg2 = d_point2trilist[parentptidx];

		if (cudamesh_segsegadjacent(parentseg1.id, parentseg2.id, d_seg2parentidxlist, d_segparentendpointidxlist))
		{
			rp = d_pointradius[parentptidx];
			if (rv < rp)
				rv = rp; // The relaxed insertion radius of new point
		}
	}
	else if (parenttype == FREEFACETVERTEX)
	{
		trihandle parentseg, parentsh;
		parentseg = d_point2trilist[newptidx];
		parentsh = d_point2trilist[parentptidx];
		if (cudamesh_segfacetadjacent(parentseg.id, parentsh.id, d_seg2parentidxlist, d_segparentendpointidxlist,
			d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
		{
			rp = d_pointradius[parentptidx];
			if (rv < rp)
				rv = rp; // The relaxed insertion radius of new point
		}

	}
	d_pointradius[newptidx] = rv;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int newptidx = oldpointsize + pos;
	int parentptidx = d_parentpt[threadId];

	REAL rv = d_smlen[threadId], rp;
	verttype parenttype = d_pointtypelist[parentptidx];

	if (parenttype == FREESEGVERTEX)
	{
		trihandle parentseg, parentsh;
		parentseg = d_point2trilist[parentptidx];
		parentsh = d_point2trilist[newptidx];

		if (cudamesh_segfacetadjacent(parentseg.id, parentsh.id, d_seg2parentidxlist, d_segparentendpointidxlist,
			d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
		{
			rp = d_pointradius[parentptidx];
			if (rv < (sqrt(2.0) * rp))
				rv = sqrt(2.0) * rp; // The relaxed insertion radius of new point
		}
	}
	else if (parenttype == FREEFACETVERTEX)
	{
		trihandle parentsh1, parentsh2;
		parentsh1 = d_point2trilist[parentptidx];
		parentsh2 = d_point2trilist[newptidx];
		if (cudamesh_facetfacetadjacent(parentsh1.id, parentsh2.id,
			d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
		{
			rp = d_pointradius[parentptidx];
			if (rv < rp)
				rv = rp; // The relaxed insertion radius of new point
		}

	}
	d_pointradius[newptidx] = rv;
}

__global__ void kernelUpdateInsertRadius_Tet(
	int* d_insertidxlist,
	REAL* d_insertptlist,
	int* d_threadlist,
	REAL* d_pointlist,
	REAL* d_pointradius,
	int* d_tetlist,
	int oldpointsize,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int splittetid = d_insertidxlist[threadId];
	tethandle splittet(splittetid, 11);

	int newptidx = oldpointsize + pos;
	REAL *newpt = cudamesh_id2pointlist(newptidx, d_pointlist);

	int orgidx = cudamesh_org(splittet, d_tetlist);
	REAL *org = cudamesh_id2pointlist(orgidx, d_pointlist);

	REAL rv = cudamesh_distance(newpt, org);
	d_pointradius[newptidx] = rv;
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int i, p;
	bool flag = false;
	trihandle neighseg, neighsh;
	tethandle neightet;

	verttype pointtype = d_pointtypelist[pos];

	if (pointtype == FREESEGVERTEX)
	{
		neighseg = d_point2trilist[pos];
		if (neighseg.id != -1)
		{
			if (d_segstatus[neighseg.id].isEmpty())
			{
				printf("Point #%d: Empty subseg neighbor #%d\n", pos, neighseg.id);
			}
			else
			{
				for (i = 0; i < 3; i++)
				{
					p = d_seglist[3 * neighseg.id + i];
					if (i == 2 && p != -1)
					{
						printf("Point #%d: Wrong point type (on subseg) or neighbor type (subseg) #%d - %d, %d, %d\n", pos,
							neighseg.id, d_seglist[3 * neighseg.id + 0], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
					}
					if (p == pos)
					{
						flag = true;
						break;
					}
				}
				if (!flag)
					printf("Point #%d: Wrong subface neighbor #%d - %d, %d, %d\n", pos,
						neighseg.id, d_seglist[3 * neighseg.id + 0], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
			}
		}
		else
		{
			printf("Point #%d: Missing segment neighbor\n");
		}
	}
	else if (pointtype == FREEFACETVERTEX)
	{
		neighsh = d_point2trilist[pos];
		if (neighsh.id != -1)
		{
			if (d_tristatus[neighsh.id].isEmpty())
			{
				printf("Point #%d: Empty subface neighbor #%d\n", pos, neighsh.id);
			}
			else
			{
				for (i = 0; i < 3; i++)
				{
					p = d_trifacelist[3 * neighsh.id + i];
					if (p == -1)
					{
						printf("Point #%d: Wrong point type (on subface) or neighbor type (subface) #%d - %d, %d, %d\n",pos,
							neighsh.id, d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					}
					if (p == pos)
					{
						flag = true;
						break;
					}
				}
				if (!flag)
					printf("Point #%d: Wrong subface neighbor #%d - %d, %d, %d\n", pos,
						neighsh.id, d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
			}
		}
		else
		{
			printf("Point #%d: Missing subface neighbor\n");
		}
	}

	neightet = d_point2tetlist[pos];
	if (neightet.id != -1)
	{
		//printf("%d ", neightet.id);
		if (d_tetstatus[neightet.id].isEmpty())
		{
			printf("Point #%d: Empty tet neighbor #%d\n", pos, neightet.id);
		}
		else
		{
			for (i = 0; i < 4; i++)
			{
				p = d_tetlist[4 * neightet.id + i];
				if (p == pos)
				{
					flag = true;
					break;
				}
			}
			if (!flag)
				printf("Point #%d: Wrong tet neighbor #%d - %d, %d, %d, %d\n", pos,
					neightet.id,
					d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_segstatus[pos].isEmpty())
		return;

	trihandle checkseg(pos, 0), neighsh, neighseg, prevseg, nextseg;
	int pa, pb, pc, pd;

	cudamesh_spivot(checkseg, neighsh, d_seg2trilist);
	if (neighsh.id != -1)
	{
		if (d_tristatus[neighsh.id].isEmpty())
		{
			printf("Subseg #%d: Empty subface neighbor #%d\n", checkseg.id, neighsh.id);
		}
		else
		{
			if (d_trifacelist[3 * neighsh.id + 2] == -1)
			{
				printf("Subseg #%d: Wrong neighbor type (Should be subface) #%d\n", checkseg.id, neighsh.id);
			}
			else
			{
				cudamesh_sspivot(neighsh, neighseg, d_tri2seglist);
				if (neighseg.id != checkseg.id)
					printf("Subseg #%d: Wrong subface neighbor #%d - %d, %d, %d\n", checkseg.id,
						neighsh.id, d_tri2seglist[3 * neighsh.id + 0].id, d_tri2seglist[3 * neighsh.id + 1].id, d_tri2seglist[3 * neighsh.id + 2].id);
				else
				{
					pa = cudamesh_sorg(checkseg, d_seglist);
					pb = cudamesh_sdest(checkseg, d_seglist);
					pc = cudamesh_sorg(neighsh, d_trifacelist);
					pd = cudamesh_sdest(neighsh, d_trifacelist);
					if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
					{

					}
					else
					{
						printf("Subseg #%d - %d, %d: Wrong subface neighbor endpoints #%d - %d, %d, %d\n", checkseg.id,
							d_seglist[3 * checkseg.id], d_seglist[3 * checkseg.id + 1],
							neighsh.id, d_trifacelist[3 * neighsh.id], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					}
				}
			}
		}
	}

	cudamesh_senextself(checkseg);
	cudamesh_spivot(checkseg, prevseg, d_seg2trilist);
	if (prevseg.id != -1)
	{
		if (d_segstatus[prevseg.id].isEmpty())
		{
			printf("Subseg #%d: Empty subseg neighbor #%d\n", checkseg.id, prevseg.id);
		}
		else
		{
			if (d_seglist[3 * prevseg.id + 2] != -1)
			{
				printf("Subseg #%d: Wrong neighbor type (Should be subseg) #%d\n", checkseg.id, prevseg.id);
			}
			else
			{
				cudamesh_spivot(prevseg, neighseg, d_seg2trilist);
				if(neighseg.id != checkseg.id)
					printf("Subseg #%d: Wrong subseg neighbor #%d - %d, %d, %d\n", checkseg.id,
						prevseg.id, d_seg2trilist[3 * prevseg.id + 0].id, d_seg2trilist[3 * prevseg.id + 1].id, d_seg2trilist[3 * prevseg.id + 2].id);
			}
		}
	}

	cudamesh_senextself(checkseg);
	cudamesh_spivot(checkseg, nextseg, d_seg2trilist);
	if (nextseg.id != -1)
	{
		if (d_segstatus[nextseg.id].isEmpty())
		{
			printf("Subseg #%d: Empty subseg neighbor #%d\n", checkseg.id, prevseg.id);
		}
		else
		{
			if (d_seglist[3 * nextseg.id + 2] != -1)
			{
				printf("Subseg #%d: Wrong neighbor type (Should be subseg) #%d\n", checkseg.id, nextseg.id);
			}
			else
			{
				cudamesh_spivot(nextseg, neighseg, d_seg2trilist);
				if (neighseg.id != checkseg.id)
					printf("Subseg #%d: Wrong subseg neighbor #%d - %d, %d, %d\n", checkseg.id,
						nextseg.id, d_seg2trilist[3 * nextseg.id + 0].id, d_seg2trilist[3 * nextseg.id + 1].id, d_seg2trilist[3 * nextseg.id + 2].id);
			}
		}
	}

	tethandle neightet;
	checkseg.shver = 0;
	cudamesh_sstpivot1(checkseg, neightet, d_seg2tetlist);
	if (neightet.id != -1)
	{
		if (d_tetstatus[neightet.id].isEmpty())
		{
			printf("Subseg #%d: Empty tet neighbor #%d\n", checkseg.id, neightet.id);
		}
		else
		{
			cudamesh_tsspivot1(neightet, neighseg, d_tet2seglist);
			if (neighseg.id != checkseg.id)
				printf("Subseg #%d: Wrong tet neighbor #%d - %d, %d, %d, %d, %d, %d\n", checkseg.id,
					neightet.id, d_tet2seglist[6 * neightet.id + 0].id, d_tet2seglist[6 * neightet.id + 1].id, d_tet2seglist[6 * neightet.id + 2].id,
					d_tet2seglist[6 * neightet.id + 3].id, d_tet2seglist[6 * neightet.id + 4].id, d_tet2seglist[6 * neightet.id + 5].id);
			else
			{
				pa = cudamesh_sorg(checkseg, d_seglist);
				pb = cudamesh_sdest(checkseg, d_seglist);
				pc = cudamesh_org(neightet, d_tetlist);
				pd = cudamesh_dest(neightet, d_tetlist);
				if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
				{

				}
				else
				{
					printf("Subseg #%d - %d, %d: Wrong tet neighbor endpoints #%d(%d) - %d, %d, %d, %d\n", checkseg.id,
						d_seglist[3 * checkseg.id], d_seglist[3 * checkseg.id + 1],
						neightet.id, neightet.ver,
						d_tetlist[4 * neightet.id], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
				}
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_tristatus[pos].isEmpty())
		return;

	trihandle checksh(pos, 0), neighseg, neighsh, neineighsh;
	tethandle neightet;
	int i, pa, pb, pc, pd, pe, pf;

	for (i = 0; i < 3; i++)
	{
		cudamesh_senextself(checksh);
		cudamesh_sspivot(checksh, neighseg, d_tri2seglist);
		if (neighseg.id != -1)
		{
			if (d_segstatus[neighseg.id].isEmpty())
				printf("Subface #%d: Empty subseg neighbor #%d\n", checksh.id, neighseg.id);
			else
			{
				cudamesh_spivot(neighseg, neighsh, d_seg2trilist);
				if (neighsh.id == -1)
				{
					printf("Subface #%d: Wrong subseg neighbor, Subface #%d - %d, %d, %d, Subseg #%d - (-1)\n",
						checksh.id, d_tri2seglist[3 * checksh.id + 0].id, d_tri2seglist[3 * checksh.id + 1].id, d_tri2seglist[3 * checksh.id + 2].id,
						neighseg.id);
				}
				else
				{
					//printf("%d ", neighsh.id);
					bool found = false;
					cudamesh_spivot(neighsh, neineighsh, d_tri2trilist);
					if (neighsh.id == checksh.id)
						found = true;
					if (neineighsh.id == -1) // this only happen when neighsh is a single subface
					{
						if(checksh.id != neighsh.id)
							printf("Subface: Wrong single subface neighbor - Checksh #%d, Neighseg #%d, Neighsh #%d\n", checksh.id, neighseg.id, neighsh.id);
					}
					else
					{
						if (neighsh.id == neineighsh.id)
						{
							if (checksh.id != neighsh.id)
								printf("Subface: Wrong single subface neighbor - Checksh #%d, Neighsh #%d, neineighsh #%d\n", checksh.id, neighsh.id, neineighsh.id);
						}
						else
						{
							while (neineighsh.id != neighsh.id)
							{
								if (neineighsh.id == checksh.id)
								{
									found = true;
									break;
								}
								cudamesh_spivotself(neineighsh, d_tri2trilist);
							}
						}
					}
					if (!found)
						printf("Subface #%d: Wrong subseg neighbor #%d, missing in loop\n",
							checksh.id, neighseg.id);
					else
					{
						pa = cudamesh_sorg(checksh, d_trifacelist);
						pb = cudamesh_sdest(checksh, d_trifacelist);
						pc = cudamesh_sorg(neighseg, d_seglist);
						pd = cudamesh_sdest(neighseg, d_seglist);
						if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
						{

						}
						else
						{
							printf("Subface #%d - %d, %d, %d: Wrong subseg neighbor endpoints #%d - %d, %d, %d\n",
								checksh.id, d_trifacelist[3 * checksh.id + 0], d_trifacelist[3 * checksh.id + 1], d_trifacelist[3 * checksh.id + 2],
								neighseg.id, d_seglist[3 * neighseg.id + 0], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
						}
					}
				}
			}
		}
	}

	for (i = 0; i < 3; i++)
	{
		cudamesh_senextself(checksh);
		cudamesh_spivot(checksh, neighsh, d_tri2trilist);
		if (neighsh.id != -1)
		{
			while (neighsh.id != checksh.id)
			{
				if (d_tristatus[neighsh.id].isEmpty())
				{
					printf("Subface #%d - %d, %d, %d - %d, %d, %d: Empty subface neighbor #%d - %d, %d, %d - %d, %d, %d\n",
						checksh.id, d_tri2trilist[3 * checksh.id + 0].id, d_tri2trilist[3 * checksh.id + 1].id, d_tri2trilist[3 * checksh.id + 2].id,
						d_trifacelist[3 * checksh.id + 0], d_trifacelist[3 * checksh.id + 1], d_trifacelist[3 * checksh.id + 2],
						neighsh.id, d_tri2trilist[3 * neighsh.id + 0].id, d_tri2trilist[3 * neighsh.id + 1].id, d_tri2trilist[3 * neighsh.id + 2].id,
						d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					break;
				}
				cudamesh_spivotself(neighsh, d_tri2trilist);
			}
		}
	}

	for (i = 0; i < 2; i++)
	{
		cudamesh_sesymself(checksh);
		cudamesh_stpivot(checksh, neightet, d_tri2tetlist);
		if (neightet.id != -1)
		{
			if (d_tetstatus[neightet.id].isEmpty())
			{
				printf("Subface #%d: Empty tet neighbor #%d\n", checksh.id, neightet.id);
			}
			else
			{
				cudamesh_tspivot(neightet, neighsh, d_tet2trilist);
				if (neighsh.id != checksh.id)
					printf("Subface #%d: Wrong tet neighbor #%d - %d, %d, %d, %d\n", checksh.id,
						neightet.id, d_tet2trilist[4 * neightet.id + 0].id, d_tet2trilist[4 * neightet.id + 1].id, d_tet2trilist[4 * neightet.id + 2].id, d_tet2trilist[4 * neightet.id + 3].id);
				else
				{
					pa = cudamesh_sorg(checksh, d_trifacelist);
					pb = cudamesh_sdest(checksh, d_trifacelist);
					pc = cudamesh_sapex(checksh, d_trifacelist);
					pd = cudamesh_org(neightet, d_tetlist);
					pe = cudamesh_dest(neightet, d_tetlist);
					pf = cudamesh_apex(neightet, d_tetlist);
					if (pa == pd && pb == pe && pc == pf)
					{

					}
					else
					{
						printf("Subface #%d - %d, %d, %d: Wrong tet neighbor endpoints #%d - %d, %d, %d, %d\n",
							checksh.id, d_trifacelist[3 * checksh.id + 0], d_trifacelist[3 * checksh.id + 1], d_trifacelist[3 * checksh.id + 2],
							neightet.id, d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
					}
				}
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_tetstatus[pos].isEmpty())
		return;

	tethandle neightet, neineightet;
	trihandle neighsh, neighseg;
	int i, pa, pb, pc, pd, pe, pf;

	for (i = 0; i < 4; i++)
	{
		neightet = d_neighborlist[4 * pos + i];
		if (neightet.id != -1)
		{
			if (d_tetstatus[neightet.id].isEmpty())
			{
				printf("Tet #%d - %d, %d, %d, %d: Empty tet neighbor #%d - %d, %d, %d, %d\n",
					pos, d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
					neightet.id, d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
			}
			else
			{
				cudamesh_fsym(neightet, neineightet, d_neighborlist);
				if (neineightet.id != pos)
					printf("Tet #%d: Wrong tet neighbor #%d - %d, %d, %d, %d\n", pos,
						neightet.id, d_neighborlist[4 * neightet.id + 0].id, d_neighborlist[4 * neightet.id + 1].id, 
						d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
				else
				{
					pa = cudamesh_org(neightet, d_tetlist);
					pb = cudamesh_dest(neightet, d_tetlist);
					pc = cudamesh_apex(neightet, d_tetlist);
					pd = cudamesh_org(neineightet, d_tetlist);
					pe = cudamesh_dest(neineightet, d_tetlist);
					pf = cudamesh_apex(neineightet, d_tetlist);
					if (pa == pe && pb == pd && pc == pf)
					{

					}
					else
					{
						printf("Tet #%d - %d, %d, %d, %d: Wrong tet neighbor endpoints #%d - %d, %d, %d, %d\n",
							pos, d_tetlist[4 * pos], d_tetlist[4 * pos + 1], d_tetlist[4 * pos + 2], d_tetlist[4 * pos + 3],
							neightet.id, d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3]);
					}
				}
			}
		}
		else
		{
			printf("Tet #%d - %d, %d, %d, %d: Empty tet neighbor #%d - %d, %d, %d, %d\n",
				pos, d_neighborlist[4 * pos].id, d_neighborlist[4 * pos + 1].id, d_neighborlist[4 * pos + 2].id, d_neighborlist[4 * pos + 3].id,
				neightet.id, d_neighborlist[4 * neightet.id].id, d_neighborlist[4 * neightet.id + 1].id, d_neighborlist[4 * neightet.id + 2].id, d_neighborlist[4 * neightet.id + 3].id);
		}
	}

	for (i = 0; i < 4; i++)
	{
		neighsh = d_tet2trilist[4 * pos + i];
		if (neighsh.id != -1)
		{
			if (d_tristatus[neighsh.id].isEmpty())
			{
				printf("Tet #%d - %d, %d, %d, %d: Empty subface neighbor #%d - %d, %d\n",
					pos, d_tet2trilist[4 * pos].id, d_tet2trilist[4 * pos + 1].id, d_tet2trilist[4 * pos + 2].id, d_tet2trilist[4 * pos + 3].id,
					neighsh.id, d_tri2tetlist[2 * neightet.id].id, d_tri2tetlist[2 * neightet.id + 1].id);
			}
			else
			{
				cudamesh_stpivot(neighsh, neightet, d_tri2tetlist);
				if(neightet.id != pos)
					printf("Tet #%d: Wrong subface neighbor #%d - %d, %d\n", pos,
						neighsh.id, d_tri2tetlist[2 * neighsh.id + 0].id, d_tri2tetlist[2 * neighsh.id + 1].id);
				else
				{
					pa = cudamesh_sorg(neighsh, d_trifacelist);
					pb = cudamesh_sdest(neighsh, d_trifacelist);
					pc = cudamesh_sapex(neighsh, d_trifacelist);
					pd = cudamesh_org(neightet, d_tetlist);
					pe = cudamesh_dest(neightet, d_tetlist);
					pf = cudamesh_apex(neightet, d_tetlist);
					if(pa == pd && pb == pe && pc == pf)
					{

					}
					else
					{
						printf("Tet #%d - %d, %d, %d, %d: Wrong subface neighbor endpoints #%d - %d, %d, %d\n", 
							pos, d_tetlist[4 * pos], d_tetlist[4 * pos + 1], d_tetlist[4 * pos + 2], d_tetlist[4 * pos + 3],
							neighsh.id, d_trifacelist[3 * neighsh.id + 0], d_trifacelist[3 * neighsh.id + 1], d_trifacelist[3 * neighsh.id + 2]);
					}
				}
			}
		}
	}

	for (i = 0; i < 6; i++)
	{
		neighseg = d_tet2seglist[6 * pos + i];
		if (neighseg.id != -1)
		{
			if(d_segstatus[neighseg.id].isEmpty())
			{
				printf("Tet #%d - %d, %d, %d, %d, %d, %d: Empty subseg neighbor #%d - %d\n",
					pos, d_tet2seglist[6 * pos].id, d_tet2seglist[6 * pos + 1].id, d_tet2seglist[6 * pos + 2].id, 
					d_tet2seglist[6 * pos + 3].id, d_tet2seglist[6 * pos + 4].id, d_tet2seglist[6 * pos + 5].id,
					neighseg.id, d_seg2tetlist[neighseg.id].id);
			}
			else
			{
				cudamesh_sstpivot1(neighseg, neightet, d_seg2tetlist);
				if (neightet.id == -1)
					printf("Tet #%d - Incident Subseg #%d has empty tet neighbor\n",
						pos, neighseg.id);
				else
				{
					pa = cudamesh_sorg(neighseg, d_seglist);
					pb = cudamesh_sdest(neighseg, d_seglist);
					pc = cudamesh_org(neightet, d_tetlist);
					pd = cudamesh_dest(neightet, d_tetlist);
					if ((pa == pc && pb == pd) || (pa == pd && pb == pc))
					{

					}
					else
					{
						printf("pa = %d, pb = %d, pc = %d, pd = %d\n", pa, pb, pc, pd);
						printf("Tet #%d(%d) - %d, %d, %d, %d: Wrong subseg neighbor endpoints #%d - %d, %d, %d\n",
							neightet.id, neightet.ver,
							d_tetlist[4 * neightet.id + 0], d_tetlist[4 * neightet.id + 1], d_tetlist[4 * neightet.id + 2], d_tetlist[4 * neightet.id + 3],
							neighseg.id, d_seglist[3 * neighseg.id], d_seglist[3 * neighseg.id + 1], d_seglist[3 * neighseg.id + 2]);
					}
				}
			}
		}
	}
}

// Split bad elements
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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (pos < numofencsegs)
	{
		if(d_threadmarker[pos] != 0)
			printf("threadId #%d - seg #%d - wrong thread marker %d\n", pos, d_badeleidlist[pos], d_threadmarker[pos]);
		else if(d_segencmarker[d_badeleidlist[pos]] < 0)
			printf("threadId #%d - seg #%d - wrong encroachement marker %d\n", pos, d_badeleidlist[pos], d_segencmarker[d_badeleidlist[pos]]);
	}
	else if (pos < numofencsubfaces + numofencsegs)
	{	
		if (d_threadmarker[pos] != 1)
			printf("threadId #%d - subface #%d - wrong thread marker %d\n", pos, d_badeleidlist[pos], d_threadmarker[pos]);
		else if (d_subfaceencmarker[d_badeleidlist[pos]] < 0)
			printf("threadId #%d - subface #%d - wrong encroachement marker %d\n", pos, d_badeleidlist[pos], d_subfaceencmarker[d_badeleidlist[pos]]);
	}
	else
	{
		if (d_threadmarker[pos] != 2)
			printf("threadId #%d - tet #%d - wrong thread marker %d\n", pos, d_badeleidlist[pos], d_threadmarker[pos]);
		else if (!d_tetstatus[d_badeleidlist[pos]].isBad() || d_tetstatus[d_badeleidlist[pos]].isEmpty())
			printf("threadId #%d - tet #%d - wrong tet status\n", pos, d_badeleidlist[pos]);
	}

}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int eleidx = d_insertidxlist[pos];
	int threadmarker = d_threadmarker[pos];

	REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

	if (threadmarker == 0) // is a subsegment
	{
		trihandle seg(eleidx, 0);
		REAL* ei = cudamesh_id2pointlist(cudamesh_sorg(seg, d_seglist), d_pointlist);
		REAL* ej = cudamesh_id2pointlist(cudamesh_sdest(seg, d_seglist), d_pointlist);

		REAL len = cudamesh_distance(ei, ej);
		d_priority[pos] = 1 / len;

		int adjflag = 0, i;

		int refptidx = d_segencmarker[eleidx];
		if (refptidx != MAXINT)
		{
			REAL* refpt = cudamesh_id2pointlist(refptidx, d_pointlist);
			REAL L, L1, t;

			if (d_pointtypelist[refptidx] == FREESEGVERTEX) {
				trihandle parentseg;
				parentseg = d_point2trilist[refptidx];
				int sidx1 = d_seg2parentlist[parentseg.id];
				int idx_pi = d_segparentlist[sidx1 * 2];
				int idx_pj = d_segparentlist[sidx1 * 2 + 1];
				REAL* far_pi = cudamesh_id2pointlist(idx_pi, d_pointlist);
				REAL* far_pj = cudamesh_id2pointlist(idx_pj, d_pointlist);

				int sidx2 = d_seg2parentlist[seg.id];
				int idx_ei = d_segparentlist[sidx2 * 2];
				int idx_ej = d_segparentlist[sidx2 * 2 + 1];
				REAL* far_ei = cudamesh_id2pointlist(idx_ei, d_pointlist);
				REAL* far_ej = cudamesh_id2pointlist(idx_ej, d_pointlist);

				if ((idx_pi == idx_ei) || (idx_pj == idx_ei)) {
					// Create a Steiner point at the intersection of the segment
					//   [far_ei, far_ej] and the sphere centered at far_ei with 
					//   radius |far_ei - refpt|.
					L = cudamesh_distance(far_ei, far_ej);
					L1 = cudamesh_distance(far_ei, refpt);
					t = L1 / L;
					for (i = 0; i < 3; i++) {
						steinpt[i] = far_ei[i] + t * (far_ej[i] - far_ei[i]);
					}
					adjflag = 1;
				}
				else if ((idx_pi == idx_ej) || (idx_pj == idx_ej)) {
					L = cudamesh_distance(far_ei, far_ej);
					L1 = cudamesh_distance(far_ej, refpt);
					t = L1 / L;
					for (i = 0; i < 3; i++) {
						steinpt[i] = far_ej[i] + t * (far_ei[i] - far_ej[i]);
					}
					adjflag = 1;
				}
				else {
					// Cut the segment by the projection point of refpt.
					projectpoint2edge(refpt, ei, ej, steinpt);
				}
			}
			else {
				// Cut the segment by the projection point of refpt.
				projectpoint2edge(refpt, ei, ej, steinpt);
			}

			// Make sure that steinpt is not too close to ei and ej.
			L = cudamesh_distance(ei, ej);
			L1 = cudamesh_distance(steinpt, ei);
			t = L1 / L;
			if ((t < 0.2) || (t > 0.8)) {
				// Split the point at the middle.
				for (i = 0; i < 3; i++) {
					steinpt[i] = ei[i] + 0.5 * (ej[i] - ei[i]);
				}
			}
		}
		else
		{
			// Split the point at the middle.
			for (i = 0; i < 3; i++) {
				steinpt[i] = ei[i] + 0.5 * (ej[i] - ei[i]);
			}
		}
	}
	else if (threadmarker == 1) // is a subface
	{
		REAL *pa, *pb, *pc;
		REAL area, rd, len;
		REAL A[4][4], rhs[4], D;
		int indx[4];
		int i;

		trihandle chkfac(eleidx, 0);
		REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

		pa = cudamesh_id2pointlist(cudamesh_sorg(chkfac, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(chkfac, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(chkfac, d_trifacelist), d_pointlist);

		// Compute the coefficient matrix A (3x3).
		A[0][0] = pb[0] - pa[0];
		A[0][1] = pb[1] - pa[1];
		A[0][2] = pb[2] - pa[2]; // vector V1 (pa->pb)
		A[1][0] = pc[0] - pa[0];
		A[1][1] = pc[1] - pa[1];
		A[1][2] = pc[2] - pa[2]; // vector V2 (pa->pc)
		cudamesh_cross(A[0], A[1], A[2]); // vector V3 (V1 X V2)

		area = 0.5 * sqrt(cudamesh_dot(A[2], A[2])); // The area of [a,b,c].
		d_priority[pos] = 1 / area;

		// Compute the right hand side vector b (3x1).
		rhs[0] = 0.5 * cudamesh_dot(A[0], A[0]); // edge [a,b]
		rhs[1] = 0.5 * cudamesh_dot(A[1], A[1]); // edge [a,c]
		rhs[2] = 0.0;

		// Solve the 3 by 3 equations use LU decomposition with partial 
		//   pivoting and backward and forward substitute.
		if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
			// A degenerate triangle. 
			//printf("kernelComputeSteinerPointOnSubface: A degenerate subface. This should not happen!\n");
		}

		cudamesh_lu_solve(A, 3, indx, rhs, 0);
		steinpt[0] = pa[0] + rhs[0];
		steinpt[1] = pa[1] + rhs[1];
		steinpt[2] = pa[2] + rhs[2];
	}
	else if(threadmarker == 2) // is a tetrahedron
	{
		int tetid = eleidx;

		int ipa, ipb, ipc, ipd;
		REAL *pa, *pb, *pc, *pd;
		REAL vda[3], vdb[3], vdc[3];
		REAL vab[3], vbc[3], vca[3];
		REAL elen[6];
		REAL smlen = 0, rd;
		REAL A[4][4], rhs[4], D;
		int indx[4];
		int i;

		ipd = d_tetlist[4 * tetid + 3];
		if (ipd == -1) {
			// This should not happend
			printf("Thread #%d - Error: Try to split a hull tet #%d!\n", pos, tetid);
			return;
		}

		ipa = d_tetlist[4 * tetid + 0];
		ipb = d_tetlist[4 * tetid + 1];
		ipc = d_tetlist[4 * tetid + 2];

		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		pc = cudamesh_id2pointlist(ipc, d_pointlist);
		pd = cudamesh_id2pointlist(ipd, d_pointlist);

		// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
		// Set the matrix A = [vda, vdb, vdc]^T.
		for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
		for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
		for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

		// Get the other edge vectors.
		for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
		for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
		for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

		if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
			// This should not happend
			//printf("Thread #%d - Error: Try to split a degenerated tet #%d!\n", threadId, tetid);
			d_tetstatus[tetid].setAbortive(true);
			d_threadmarker[pos] = -1;
			return;
		}

		// Calculate the circumcenter and radius of this tet.
		rhs[0] = 0.5 * cudamesh_dot(vda, vda);
		rhs[1] = 0.5 * cudamesh_dot(vdb, vdb);
		rhs[2] = 0.5 * cudamesh_dot(vdc, vdc);
		cudamesh_lu_solve(A, 3, indx, rhs, 0);
		for (i = 0; i < 3; i++)
		{
			steinpt[i] = pd[i] + rhs[i];
		}

		//Calculate the shortest edge length.
		elen[0] = cudamesh_dot(vda, vda);
		elen[1] = cudamesh_dot(vdb, vdb);
		elen[2] = cudamesh_dot(vdc, vdc);
		elen[3] = cudamesh_dot(vab, vab);
		elen[4] = cudamesh_dot(vbc, vbc);
		elen[5] = cudamesh_dot(vca, vca);

		// Use volume as priority
		// Use heron-type formula to compute the volume of a tetrahedron
		// https://en.wikipedia.org/wiki/Heron%27s_formula
		REAL U, V, W, u, v, w; // first three form a triangle; u opposite to U and so on
		REAL X, x, Y, y, Z, z;
		REAL a, b, c, d;
		U = sqrt(elen[3]); //ab
		V = sqrt(elen[4]); //bc
		W = sqrt(elen[5]); //ca
		u = sqrt(elen[2]); //dc
		v = sqrt(elen[0]); //da
		w = sqrt(elen[1]); //db

		X = (w - U + v)*(U + v + w);
		x = (U - v + w)*(v - w + U);
		Y = (u - V + w)*(V + w + u);
		y = (V - w + u)*(w - u + V);
		Z = (v - W + u)*(W + u + v);
		z = (W - u + v)*(u - v + W);

		a = sqrt(x*Y*Z);
		b = sqrt(y*Z*X);
		c = sqrt(z*X*Y);
		d = sqrt(x*y*z);

		REAL vol = sqrt((-a + b + c + d)*(a - b + c + d)*(a + b - c + d)*(a + b + c - d)) / (192 * u*v*w);
		d_priority[pos] = 1 / vol;
	}
}

__global__ void kernelComputePriorities(
	REAL* d_pointlist,
	int* d_seglist,
	int* d_trifacelist,
	int* d_tetlist,
	int* d_insertidxlist,
	int* d_threadmarker,
	REAL* d_priority,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int eleidx = d_insertidxlist[pos];
	int threadmarker = d_threadmarker[pos];

	if (threadmarker == 0) // is a subsegment
	{
		trihandle seg(eleidx, 0);
		REAL* ei = cudamesh_id2pointlist(cudamesh_sorg(seg, d_seglist), d_pointlist);
		REAL* ej = cudamesh_id2pointlist(cudamesh_sdest(seg, d_seglist), d_pointlist);

		REAL len = cudamesh_distance(ei, ej);
		d_priority[pos] = len;
	}
	else if (threadmarker == 1) // is a subface
	{
		REAL *pa, *pb, *pc;
		REAL sarea;

		trihandle chkfac(eleidx, 0);

		pa = cudamesh_id2pointlist(cudamesh_sorg(chkfac, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(chkfac, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(chkfac, d_trifacelist), d_pointlist);
		sarea = cudamesh_triangle_squared_area(pa, pb, pc);
		d_priority[pos] = sarea;
	}
	else if (threadmarker == 2) // is a tetrahedron
	{
		int tetid = eleidx;

		int ipa, ipb, ipc, ipd;
		REAL *pa, *pb, *pc, *pd;

		ipd = d_tetlist[4 * tetid + 3];
		if (ipd == -1) {
			// This should not happend
			return;
		}

		ipa = d_tetlist[4 * tetid + 0];
		ipb = d_tetlist[4 * tetid + 1];
		ipc = d_tetlist[4 * tetid + 2];

		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		pc = cudamesh_id2pointlist(ipc, d_pointlist);
		pd = cudamesh_id2pointlist(ipd, d_pointlist);

		REAL vol = cudamesh_tetrahedronvolume(pa, pb, pc, pd);
		d_priority[pos] = vol;
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int eleidx = d_insertidxlist[pos];
	int threadmarker = d_threadmarker[pos];
	if (threadmarker < 0)
		return;

	REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

	if (threadmarker == 0) // is a subsegment
	{
		trihandle seg(eleidx, 0);
		REAL* ei = cudamesh_id2pointlist(cudamesh_sorg(seg, d_seglist), d_pointlist);
		REAL* ej = cudamesh_id2pointlist(cudamesh_sdest(seg, d_seglist), d_pointlist);

		REAL len = cudamesh_distance(ei, ej);

		int adjflag = 0, i;

		int refptidx = d_segencmarker[eleidx];
		if (refptidx != MAXINT)
		{
			REAL* refpt = cudamesh_id2pointlist(refptidx, d_pointlist);
			REAL L, L1, t;

			if (d_pointtypelist[refptidx] == FREESEGVERTEX) {
				trihandle parentseg;
				parentseg = d_point2trilist[refptidx];
				int sidx1 = d_seg2parentlist[parentseg.id];
				int idx_pi = d_segparentlist[sidx1 * 2];
				int idx_pj = d_segparentlist[sidx1 * 2 + 1];
				REAL* far_pi = cudamesh_id2pointlist(idx_pi, d_pointlist);
				REAL* far_pj = cudamesh_id2pointlist(idx_pj, d_pointlist);

				int sidx2 = d_seg2parentlist[seg.id];
				int idx_ei = d_segparentlist[sidx2 * 2];
				int idx_ej = d_segparentlist[sidx2 * 2 + 1];
				REAL* far_ei = cudamesh_id2pointlist(idx_ei, d_pointlist);
				REAL* far_ej = cudamesh_id2pointlist(idx_ej, d_pointlist);

				if ((idx_pi == idx_ei) || (idx_pj == idx_ei)) {
					// Create a Steiner point at the intersection of the segment
					//   [far_ei, far_ej] and the sphere centered at far_ei with 
					//   radius |far_ei - refpt|.
					L = cudamesh_distance(far_ei, far_ej);
					L1 = cudamesh_distance(far_ei, refpt);
					t = L1 / L;
					for (i = 0; i < 3; i++) {
						steinpt[i] = far_ei[i] + t * (far_ej[i] - far_ei[i]);
					}
					adjflag = 1;
				}
				else if ((idx_pi == idx_ej) || (idx_pj == idx_ej)) {
					L = cudamesh_distance(far_ei, far_ej);
					L1 = cudamesh_distance(far_ej, refpt);
					t = L1 / L;
					for (i = 0; i < 3; i++) {
						steinpt[i] = far_ej[i] + t * (far_ei[i] - far_ej[i]);
					}
					adjflag = 1;
				}
				else {
					// Cut the segment by the projection point of refpt.
					projectpoint2edge(refpt, ei, ej, steinpt);
				}
			}
			else {
				// Cut the segment by the projection point of refpt.
				projectpoint2edge(refpt, ei, ej, steinpt);
			}

			// Make sure that steinpt is not too close to ei and ej.
			L = cudamesh_distance(ei, ej);
			L1 = cudamesh_distance(steinpt, ei);
			t = L1 / L;
			if ((t < 0.2) || (t > 0.8)) {
				// Split the point at the middle.
				for (i = 0; i < 3; i++) {
					steinpt[i] = ei[i] + 0.5 * (ej[i] - ei[i]);
				}
			}
		}
		else
		{
			// Split the point at the middle.
			for (i = 0; i < 3; i++) {
				steinpt[i] = ei[i] + 0.5 * (ej[i] - ei[i]);
			}
		}
	}
	else if (threadmarker == 1) // is a subface
	{
		REAL *pa, *pb, *pc;
		REAL rd, len;
		REAL A[4][4], rhs[4], D;
		int indx[4];
		int i;

		trihandle chkfac(eleidx, 0);
		REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

		pa = cudamesh_id2pointlist(cudamesh_sorg(chkfac, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(chkfac, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(chkfac, d_trifacelist), d_pointlist);

		// Compute the coefficient matrix A (3x3).
		A[0][0] = pb[0] - pa[0];
		A[0][1] = pb[1] - pa[1];
		A[0][2] = pb[2] - pa[2]; // vector V1 (pa->pb)
		A[1][0] = pc[0] - pa[0];
		A[1][1] = pc[1] - pa[1];
		A[1][2] = pc[2] - pa[2]; // vector V2 (pa->pc)
		cudamesh_cross(A[0], A[1], A[2]); // vector V3 (V1 X V2)

		// Compute the right hand side vector b (3x1).
		rhs[0] = 0.5 * cudamesh_dot(A[0], A[0]); // edge [a,b]
		rhs[1] = 0.5 * cudamesh_dot(A[1], A[1]); // edge [a,c]
		rhs[2] = 0.0;

		// Solve the 3 by 3 equations use LU decomposition with partial 
		//   pivoting and backward and forward substitute.
		if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
			// A degenerate triangle. 
			//printf("kernelComputeSteinerPointOnSubface: A degenerate subface. This should not happen!\n");
		}

		cudamesh_lu_solve(A, 3, indx, rhs, 0);
		steinpt[0] = pa[0] + rhs[0];
		steinpt[1] = pa[1] + rhs[1];
		steinpt[2] = pa[2] + rhs[2];
	}
	else if (threadmarker == 2) // is a tetrahedron
	{
		int tetid = eleidx;

		int ipa, ipb, ipc, ipd;
		REAL *pa, *pb, *pc, *pd;
		REAL vda[3], vdb[3], vdc[3];
		REAL vab[3], vbc[3], vca[3];
		REAL elen[6];
		REAL smlen = 0, rd;
		REAL A[4][4], rhs[4], D;
		int indx[4];
		int i;

		ipd = d_tetlist[4 * tetid + 3];
		if (ipd == -1) {
			// This should not happend
			//printf("Thread #%d - Error: Try to split a hull tet #%d!\n", pos, tetid);
			return;
		}

		ipa = d_tetlist[4 * tetid + 0];
		ipb = d_tetlist[4 * tetid + 1];
		ipc = d_tetlist[4 * tetid + 2];

		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		pc = cudamesh_id2pointlist(ipc, d_pointlist);
		pd = cudamesh_id2pointlist(ipd, d_pointlist);

		// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
		// Set the matrix A = [vda, vdb, vdc]^T.
		for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
		for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
		for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

		// Get the other edge vectors.
		for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
		for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
		for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

		if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
			// This should not happend
			//printf("Thread #%d - Error: Try to split a degenerated tet #%d!\n", threadId, tetid);
			d_tetstatus[tetid].setAbortive(true);
			d_threadmarker[pos] = -1;
			return;
		}

		// Calculate the circumcenter and radius of this tet.
		rhs[0] = 0.5 * cudamesh_dot(vda, vda);
		rhs[1] = 0.5 * cudamesh_dot(vdb, vdb);
		rhs[2] = 0.5 * cudamesh_dot(vdc, vdc);
		cudamesh_lu_solve(A, 3, indx, rhs, 0);
		for (i = 0; i < 3; i++)
		{
			steinpt[i] = pd[i] + rhs[i];
		}
	}
}

__global__ void kernelModifyPriority(
	REAL* d_priorityreal,
	int* d_priorityint,
	REAL offset0,
	REAL offset1,
	REAL offset2,
	int* d_threadmarker,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadmarker = d_threadmarker[pos];
	REAL offset;
	if (threadmarker == 0)
		offset = offset0;
	else if (threadmarker == 1)
		offset = offset1;
	else
		offset = offset2;
	REAL priority = d_priorityreal[pos] + offset;
	d_priorityreal[pos] = priority;
	d_priorityint[pos] = __float_as_int((float)priority);
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadmarker = d_threadmarker[pos];
	int eleidx = d_insertidxlist[pos];

	if (threadmarker == 0)
	{
		int segId = eleidx;
		if (d_segstatus[segId].isAbortive())
		{
			d_threadmarker[pos] = -1;
			return;
		}

		int encptidx = d_segencmarker[pos];
		if (encptidx != MAXINT) // not encroached by splitting segment and subface routines
			return;

		trihandle splitseg(segId, 0);
		int ipa, ipb;
		ipa = cudamesh_sorg(splitseg, d_seglist);
		ipb = cudamesh_sdest(splitseg, d_seglist);
		REAL *pa, *pb;
		pa = cudamesh_id2pointlist(ipa, d_pointlist);
		pb = cudamesh_id2pointlist(ipb, d_pointlist);
		REAL len = cudamesh_distance(pa, pb);
		REAL smrrv = d_pointradius[ipa];
		REAL rrv = d_pointradius[ipb];
		if (rrv > 0)
		{
			if (smrrv > 0)
			{
				if (rrv < smrrv)
				{
					smrrv = rrv;
				}
			}
			else
			{
				smrrv = rrv;
			}
		}
		if (smrrv > 0)
		{
			if ((fabs(smrrv - len) / len) < EPSILON)
				smrrv = len;
			if (len < smrrv)
			{
				d_segstatus[segId].setAbortive(true);
				d_threadmarker[pos] = -1;
				return;
			}
		}
	}
	else if (threadmarker == 1)
	{
		int subfaceid = eleidx;
		if (d_tristatus[subfaceid].isAbortive())
		{
			d_threadmarker[pos] = -1;
			return;
		}

		int encptidx = d_subfaceencmarker[subfaceid];
		if (encptidx == MAXINT) // Mark as encroached when trying to split a tet
			return;

		trihandle parentseg, parentsh;
		trihandle splitfac(subfaceid, 0);
		REAL rv, rp;
		REAL* newpt = d_steinerptlist + 3 * pos;
		REAL* encpt = cudamesh_id2pointlist(encptidx, d_pointlist);

		rv = cudamesh_distance(newpt, encpt);
		if (d_pointtypelist[encptidx] == FREESEGVERTEX)
		{
			parentseg = d_point2trilist[encptidx];
			if (cudamesh_segfacetadjacent(parentseg.id, splitfac.id,
				d_seg2parentidxlist, d_segparentendpointidxlist,
				d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
			{
				rp = d_pointradius[encptidx];
				if (rv < (sqrt(2.0) * rp))
				{
					// This insertion may cause no termination.
					d_threadmarker[pos] = -1; // Reject the insertion of newpt.
					d_tristatus[subfaceid].setAbortive(true);
				}
			}
		}
		else if (d_pointtypelist[encptidx] == FREEFACETVERTEX)
		{
			parentsh = d_point2trilist[encptidx];
			if (cudamesh_facetfacetadjacent(parentsh.id, splitfac.id,
				d_tri2parentidxlist, d_triid2parentoffsetlist, d_triparentendpointidxlist))
			{
				rp = d_pointradius[encptidx];
				if (rv < rp)
				{
					d_threadmarker[pos] = -1; // Reject the insertion of newpt.
					d_tristatus[subfaceid].setAbortive(true);
				}
			}
		}
	}
	else
	{
		int tetid = eleidx;
		if (d_tetstatus[tetid].isAbortive())
		{
			d_threadmarker[pos] = -1;
			return;
		}

		tethandle chktet(tetid, 11), checkedge;
		int ie1, ie2;
		int i, j;
		REAL *e1, *e2;
		REAL smlen = 0;
		REAL rrv, smrrv;
		REAL elen[6];

		// Get the shortest edge of this tet.
		checkedge.id = chktet.id;
		for (i = 0; i < 6; i++) {
			checkedge.ver = raw_edge2ver[i];
			ie1 = cudamesh_org(checkedge, d_tetlist);
			ie2 = cudamesh_dest(checkedge, d_tetlist);
			e1 = cudamesh_id2pointlist(ie1, d_pointlist);
			e2 = cudamesh_id2pointlist(ie2, d_pointlist);
			elen[i] = cudamesh_distance(e1, e2);
			if (i == 0) {
				smlen = elen[i];
				j = 0;
			}
			else {
				if (elen[i] < smlen) {
					smlen = elen[i];
					j = i;
				}
			}
		}
		// Check if the edge is too short.
		checkedge.ver = raw_edge2ver[j];
		// Get the smallest rrv of e1 and e2.
		// Note: if rrv of e1 and e2 is zero. Do not use it.
		ie1 = cudamesh_org(checkedge, d_tetlist);
		smrrv = d_pointradius[ie1];
		ie2 = cudamesh_dest(checkedge, d_tetlist);
		rrv = d_pointradius[ie2];
		if (rrv > 0) {
			if (smrrv > 0) {
				if (rrv < smrrv) {
					smrrv = rrv;
				}
			}
			else {
				smrrv = rrv;
			}
		}
		if (smrrv > 0) {
			// To avoid rounding error, round smrrv before doing comparison.
			if ((fabs(smrrv - smlen) / smlen) <EPSILON) {
				smrrv = smlen;
			}
			if (smrrv > smlen) {
				d_tetstatus[tetid].setAbortive(true);
				d_threadmarker[pos] = -1;
				return;
			}
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	int eleidx = d_insertidxlist[threadId];

	if (threadmarker == 0)
	{
		trihandle splitseg(eleidx, 0);
		tethandle searchtet;
		cudamesh_sstpivot1(splitseg, searchtet, d_seg2tetlist);
		d_searchtet[threadId] = searchtet;
		d_pointlocation[threadId] = ONEDGE;
	}
	else if (threadmarker == 1)
	{
		int step = 1;
		int subfaceid = eleidx;
		d_searchsh[threadId] = trihandle(subfaceid, 0);

		trihandle neighsh;
		trihandle *searchsh = d_searchsh + threadId;
		REAL *searchpt = d_steinerptlist + 3 * threadId;
		REAL *pa, *pb, *pc;
		unsigned long *randomseed = d_randomseed + pos;
		REAL abvpt[3];

		// Check if coordinates are valid
		if (cudamesh_isInvalid(searchpt[0]) ||
			cudamesh_isInvalid(searchpt[1]) ||
			cudamesh_isInvalid(searchpt[2]))
		{
			d_tristatus[subfaceid].setAbortive(true);
			d_threadmarker[threadId] = -1;
			return;
		}

		enum locateresult loc;
		enum { MOVE_BC, MOVE_CA } nextmove;
		REAL ori, ori_bc, ori_ca;
		int i;

		pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

		// Calculate an above point for this facet.
		cudamesh_calculateabovepoint4(searchpt, pa, pb, pc, abvpt);

		// 'abvpt' is given. Make sure it is above [a,b,c]
		ori = cuda_orient3d(pa, pb, pc, abvpt);
		assert(ori != 0); // SELF_CHECK
		if (ori > 0) {
			cudamesh_sesymself(*searchsh); // Reverse the face orientation.
		}

		// Find an edge of the face s.t. p lies on its right-hand side (CCW).
		for (i = 0; i < 3; i++) {
			pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
			pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
			ori = cuda_orient3d(pa, pb, abvpt, searchpt);
			if (ori > 0) break;
			cudamesh_senextself(*searchsh);
		}
		assert(i < 3); // SELF_CHECK

		pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

		if (pc[0] == searchpt[0] && pc[1] == searchpt[1] && pc[2] == searchpt[2]) {
			cudamesh_senext2self(*searchsh);
			loc = ONVERTEX;
		}
		else
		{
			while (1) {
				ori_bc = cuda_orient3d(pb, pc, abvpt, searchpt);
				ori_ca = cuda_orient3d(pc, pa, abvpt, searchpt);

				if (ori_bc < 0) {
					if (ori_ca < 0) { // (--)
									  // Any of the edges is a viable move.
						if (cudamesh_randomnation(randomseed, 2)) {
							nextmove = MOVE_CA;
						}
						else {
							nextmove = MOVE_BC;
						}
					}
					else { // (-#)
						   // Edge [b, c] is viable.
						nextmove = MOVE_BC;
					}
				}
				else {
					if (ori_ca < 0) { // (#-)
									  // Edge [c, a] is viable.
						nextmove = MOVE_CA;
					}
					else {
						if (ori_bc > 0) {
							if (ori_ca > 0) { // (++)
								loc = ONFACE;  // Inside [a, b, c].
								break;
							}
							else { // (+0)
								cudamesh_senext2self(*searchsh); // On edge [c, a].
								loc = ONEDGE;
								break;
							}
						}
						else { // ori_bc == 0
							if (ori_ca > 0) { // (0+)
								cudamesh_senextself(*searchsh); // On edge [b, c].
								loc = ONEDGE;
								break;
							}
							else { // (00)
								   // p is coincident with vertex c. 
								cudamesh_senext2self(*searchsh);
								loc = ONVERTEX;
								break;
							}
						}
					}
				}

				// Move to the next face.
				if (nextmove == MOVE_BC) {
					cudamesh_senextself(*searchsh);
				}
				else {
					cudamesh_senext2self(*searchsh);
				}

				// NON-convex case. Check if we will cross a boundary.
				if (cudamesh_isshsubseg(*searchsh, d_tri2seglist)) {
					loc = ENCSEGMENT;
					break;
				}

				cudamesh_spivot(*searchsh, neighsh, d_tri2trilist);
				if (neighsh.id == -1) {
					loc = OUTSIDE; // A hull edge.
					break;
				}

				// Adjust the edge orientation.
				if (cudamesh_sorg(neighsh, d_trifacelist) != cudamesh_sdest(*searchsh, d_trifacelist)) {
					cudamesh_sesymself(neighsh);
				}
				assert(cudamesh_sorg(neighsh, d_trifacelist) == cudamesh_sdest(*searchsh, d_trifacelist)); // SELF_CHECK

																										   // Update the newly discovered face and its endpoints.
				*searchsh = neighsh;
				pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
				pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
				pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

				if (pc == searchpt) {
					cudamesh_senext2self(*searchsh);
					loc = ONVERTEX;
					break;
				}

				step++;

				//if (step > 1000) // invalid point coordinates
				//{
				//	printf("Subface %d, %d - %lf, %lf, %lf\n", eleidx, threadId, searchpt[0], searchpt[1], searchpt[2]);
				//}

			} // while (1)
		}

		d_pointlocation[threadId] = loc;
		if (!(loc == ONFACE || loc == ONEDGE))
		{
			if(numofsplittablesubsegs == 0)
				d_tristatus[subfaceid].setAbortive(true); // mark the encroached subface rather than the located one
			d_threadmarker[threadId] = -1;
			return;
		}

		tethandle searchtet;
		cudamesh_stpivot(*searchsh, searchtet, d_tri2tetlist);
		d_searchtet[threadId] = searchtet;
	}
	else
	{
		int tetid = eleidx;
		tethandle* searchtet = d_searchtet + threadId;
		REAL* searchpt = d_steinerptlist + 3 * threadId;
		unsigned long* randomseed = d_randomseed + pos;

		// Check if coordinates are valid
		if (cudamesh_isInvalid(searchpt[0]) || 
			cudamesh_isInvalid(searchpt[1]) ||
			cudamesh_isInvalid(searchpt[2]))
		{
			d_tetstatus[tetid].setAbortive(true);
			d_threadmarker[threadId] = -1;
			return;
		}

		REAL *torg, *tdest, *tapex, *toppo;
		enum { ORGMOVE, DESTMOVE, APEXMOVE } nextmove;
		REAL ori, oriorg, oridest, oriapex;
		enum locateresult loc = OUTSIDE;
		int t1ver;
		int s;
		int step = 1;

		// Init searchtet
		searchtet->id = tetid;
		searchtet->ver = 11;

		// Check if we are in the outside of the convex hull.
		if (cudamesh_ishulltet(*searchtet, d_tetlist)) {
			// Get its adjacent tet (inside the hull).
			searchtet->ver = 3;
			cudamesh_fsymself(*searchtet, d_neighborlist);
		}

		// Let searchtet be the face such that 'searchpt' lies above to it.
		for (searchtet->ver = 0; searchtet->ver < 4; searchtet->ver++) {
			torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
			tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
			tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);
			ori = cuda_orient3d(torg, tdest, tapex, searchpt);
			if (ori < 0.0) break;
		}
		assert(searchtet->ver != 4);

		// Walk through tetrahedra to locate the point.
		while (true) {

			toppo = cudamesh_id2pointlist(cudamesh_oppo(*searchtet, d_tetlist), d_pointlist);

			// Check if the vertex is we seek.
			if (toppo[0] == searchpt[0] && toppo[1] == searchpt[1] && toppo[2] == searchpt[2]) {
				// Adjust the origin of searchtet to be searchpt.
				cudamesh_esymself(*searchtet);
				cudamesh_eprevself(*searchtet);
				loc = ONVERTEX; // return ONVERTEX;
				break;
			}

			// We enter from one of serarchtet's faces, which face do we exit?
			oriorg = cuda_orient3d(tdest, tapex, toppo, searchpt);
			oridest = cuda_orient3d(tapex, torg, toppo, searchpt);
			oriapex = cuda_orient3d(torg, tdest, toppo, searchpt);

			// Now decide which face to move. It is possible there are more than one
			//   faces are viable moves. If so, randomly choose one.
			if (oriorg < 0) {
				if (oridest < 0) {
					if (oriapex < 0) {
						// All three faces are possible.
						s = cudamesh_randomnation(randomseed, 3); // 's' is in {0,1,2}.
						if (s == 0) {
							nextmove = ORGMOVE;
						}
						else if (s == 1) {
							nextmove = DESTMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Two faces, opposite to origin and destination, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(randomseed, 2)) {
							nextmove = ORGMOVE;
						}
						else {
							nextmove = DESTMOVE;
						}
					}
				}
				else {
					if (oriapex < 0) {
						// Two faces, opposite to origin and apex, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(randomseed, 2)) {
							nextmove = ORGMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Only the face opposite to origin is viable.
						nextmove = ORGMOVE;
					}
				}
			}
			else {
				if (oridest < 0) {
					if (oriapex < 0) {
						// Two faces, opposite to destination and apex, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(randomseed, 2)) {
							nextmove = DESTMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Only the face opposite to destination is viable.
						nextmove = DESTMOVE;
					}
				}
				else {
					if (oriapex < 0) {
						// Only the face opposite to apex is viable.
						nextmove = APEXMOVE;
					}
					else {
						// The point we seek must be on the boundary of or inside this
						//   tetrahedron. Check for boundary cases.
						if (oriorg == 0) {
							// Go to the face opposite to origin.
							cudamesh_enextesymself(*searchtet);
							if (oridest == 0) {
								cudamesh_eprevself(*searchtet); // edge oppo->apex
								if (oriapex == 0) {
									// oppo is duplicated with p.
									loc = ONVERTEX; // return ONVERTEX;
									break;
								}
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							if (oriapex == 0) {
								cudamesh_enextself(*searchtet); // edge dest->oppo
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							loc = ONFACE; // return ONFACE;
							break;
						}
						if (oridest == 0) {
							// Go to the face opposite to destination.
							cudamesh_eprevesymself(*searchtet);
							if (oriapex == 0) {
								cudamesh_eprevself(*searchtet); // edge oppo->org
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							loc = ONFACE; // return ONFACE;
							break;
						}
						if (oriapex == 0) {
							// Go to the face opposite to apex
							cudamesh_esymself(*searchtet);
							loc = ONFACE; // return ONFACE;
							break;
						}
						loc = INTETRAHEDRON; // return INTETRAHEDRON;
						break;
					}
				}
			}

			// Move to the selected face.
			if (nextmove == ORGMOVE) {
				cudamesh_enextesymself(*searchtet);
			}
			else if (nextmove == DESTMOVE) {
				cudamesh_eprevesymself(*searchtet);
			}
			else {
				cudamesh_esymself(*searchtet);
			}
			// Move to the adjacent tetrahedron (maybe a hull tetrahedron).
			cudamesh_fsymself(*searchtet, d_neighborlist);
			if (cudamesh_oppo(*searchtet, d_tetlist) == -1) {
				loc = OUTSIDE; // return OUTSIDE;
				break;
			}

			// Retreat the three vertices of the base face.
			torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
			tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
			tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);

			step++;

			//if (step > 1000) // Invalid point coordinates
			//{
			//	printf("Tet %d, %d - %lf, %lf, %lf\n", eleidx, threadId, searchpt[0], searchpt[1], searchpt[2]);
			//}

		} // while (true)

		d_pointlocation[threadId] = loc;

		if (loc == ONVERTEX)
		{
			d_tetstatus[tetid].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int threadmarker = d_threadmarker[threadId];
	int eleidx = d_insertidxlist[threadId];

	if (threadmarker == 0)
	{
		trihandle splitseg(eleidx, 0);
		tethandle searchtet;
		cudamesh_sstpivot1(splitseg, searchtet, d_seg2tetlist);
		d_searchtet[threadId] = searchtet;
		d_pointlocation[threadId] = ONEDGE;
	}
	else if (threadmarker == 1)
	{
		int step = 1;
		int subfaceid = eleidx;
		d_searchsh[threadId] = trihandle(subfaceid, 0);

		trihandle neighsh;
		trihandle *searchsh = d_searchsh + threadId;
		REAL *searchpt = d_steinerptlist + 3 * threadId;
		REAL *pa, *pb, *pc;
		unsigned long randomseed = 1;
		REAL abvpt[3];

		enum locateresult loc;
		enum { MOVE_BC, MOVE_CA } nextmove;
		REAL ori, ori_bc, ori_ca;
		int i;

		pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
		pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
		pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

		// Calculate an above point for this facet.
		cudamesh_calculateabovepoint4(searchpt, pa, pb, pc, abvpt);

		// 'abvpt' is given. Make sure it is above [a,b,c]
		ori = cuda_orient3d(pa, pb, pc, abvpt);
		assert(ori != 0); // SELF_CHECK
		if (ori > 0) {
			cudamesh_sesymself(*searchsh); // Reverse the face orientation.
		}

		// Find an edge of the face s.t. p lies on its right-hand side (CCW).
		for (i = 0; i < 3; i++) {
			pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
			pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
			ori = cuda_orient3d(pa, pb, abvpt, searchpt);
			if (ori > 0) break;
			cudamesh_senextself(*searchsh);
		}
		assert(i < 3); // SELF_CHECK

		pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

		if (pc[0] == searchpt[0] && pc[1] == searchpt[1] && pc[2] == searchpt[2]) {
			cudamesh_senext2self(*searchsh);
			loc = ONVERTEX;
		}
		else
		{
			while (1) {
				ori_bc = cuda_orient3d(pb, pc, abvpt, searchpt);
				ori_ca = cuda_orient3d(pc, pa, abvpt, searchpt);

				if (ori_bc < 0) {
					if (ori_ca < 0) { // (--)
									  // Any of the edges is a viable move.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = MOVE_CA;
						}
						else {
							nextmove = MOVE_BC;
						}
					}
					else { // (-#)
						   // Edge [b, c] is viable.
						nextmove = MOVE_BC;
					}
				}
				else {
					if (ori_ca < 0) { // (#-)
									  // Edge [c, a] is viable.
						nextmove = MOVE_CA;
					}
					else {
						if (ori_bc > 0) {
							if (ori_ca > 0) { // (++)
								loc = ONFACE;  // Inside [a, b, c].
								break;
							}
							else { // (+0)
								cudamesh_senext2self(*searchsh); // On edge [c, a].
								loc = ONEDGE;
								break;
							}
						}
						else { // ori_bc == 0
							if (ori_ca > 0) { // (0+)
								cudamesh_senextself(*searchsh); // On edge [b, c].
								loc = ONEDGE;
								break;
							}
							else { // (00)
								   // p is coincident with vertex c. 
								cudamesh_senext2self(*searchsh);
								loc = ONVERTEX;
								break;
							}
						}
					}
				}

				// Move to the next face.
				if (nextmove == MOVE_BC) {
					cudamesh_senextself(*searchsh);
				}
				else {
					cudamesh_senext2self(*searchsh);
				}

				// NON-convex case. Check if we will cross a boundary.
				if (cudamesh_isshsubseg(*searchsh, d_tri2seglist)) {
					loc = ENCSEGMENT;
					break;
				}

				cudamesh_spivot(*searchsh, neighsh, d_tri2trilist);
				if (neighsh.id == -1) {
					loc = OUTSIDE; // A hull edge.
					break;
				}

				// Adjust the edge orientation.
				if (cudamesh_sorg(neighsh, d_trifacelist) != cudamesh_sdest(*searchsh, d_trifacelist)) {
					cudamesh_sesymself(neighsh);
				}
				assert(cudamesh_sorg(neighsh, d_trifacelist) == cudamesh_sdest(*searchsh, d_trifacelist)); // SELF_CHECK

				// Update the newly discovered face and its endpoints.
				*searchsh = neighsh;
				pa = cudamesh_id2pointlist(cudamesh_sorg(*searchsh, d_trifacelist), d_pointlist);
				pb = cudamesh_id2pointlist(cudamesh_sdest(*searchsh, d_trifacelist), d_pointlist);
				pc = cudamesh_id2pointlist(cudamesh_sapex(*searchsh, d_trifacelist), d_pointlist);

				if (pc == searchpt) {
					cudamesh_senext2self(*searchsh);
					loc = ONVERTEX;
					break;
				}

				step++;

			} // while (1)
		}

		d_pointlocation[threadId] = loc;
		if (!(loc == ONFACE || loc == ONEDGE))
		{
			if (numofsplittablesubsegs == 0)
				d_tristatus[subfaceid].setAbortive(true); // mark the encroached subface rather than the located one
			d_threadmarker[threadId] = -1;
			return;
		}

		tethandle searchtet;
		cudamesh_stpivot(*searchsh, searchtet, d_tri2tetlist);
		d_searchtet[threadId] = searchtet;
	}
	else
	{
		int tetid = eleidx;
		tethandle* searchtet = d_searchtet + threadId;
		REAL* searchpt = d_steinerptlist + 3 * threadId;
		unsigned long randomseed = 1;

		REAL *torg, *tdest, *tapex, *toppo;
		enum { ORGMOVE, DESTMOVE, APEXMOVE } nextmove;
		REAL ori, oriorg, oridest, oriapex;
		enum locateresult loc = OUTSIDE;
		int t1ver;
		int s;
		int step = 1;

		// Init searchtet
		searchtet->id = tetid;
		searchtet->ver = 11;

		// Check if we are in the outside of the convex hull.
		if (cudamesh_ishulltet(*searchtet, d_tetlist)) {
			// Get its adjacent tet (inside the hull).
			searchtet->ver = 3;
			cudamesh_fsymself(*searchtet, d_neighborlist);
		}

		// Let searchtet be the face such that 'searchpt' lies above to it.
		for (searchtet->ver = 0; searchtet->ver < 4; searchtet->ver++) {
			torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
			tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
			tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);
			ori = cuda_orient3d(torg, tdest, tapex, searchpt);
			if (ori < 0.0) break;
		}
		assert(searchtet->ver != 4);

		// Walk through tetrahedra to locate the point.
		while (true) {

			toppo = cudamesh_id2pointlist(cudamesh_oppo(*searchtet, d_tetlist), d_pointlist);

			// Check if the vertex is we seek.
			if (toppo[0] == searchpt[0] && toppo[1] == searchpt[1] && toppo[2] == searchpt[2]) {
				// Adjust the origin of searchtet to be searchpt.
				cudamesh_esymself(*searchtet);
				cudamesh_eprevself(*searchtet);
				loc = ONVERTEX; // return ONVERTEX;
				break;
			}

			// We enter from one of serarchtet's faces, which face do we exit?
			oriorg = cuda_orient3d(tdest, tapex, toppo, searchpt);
			oridest = cuda_orient3d(tapex, torg, toppo, searchpt);
			oriapex = cuda_orient3d(torg, tdest, toppo, searchpt);

			// Now decide which face to move. It is possible there are more than one
			//   faces are viable moves. If so, randomly choose one.
			if (oriorg < 0) {
				if (oridest < 0) {
					if (oriapex < 0) {
						// All three faces are possible.
						s = cudamesh_randomnation(&randomseed, 3); // 's' is in {0,1,2}.
						if (s == 0) {
							nextmove = ORGMOVE;
						}
						else if (s == 1) {
							nextmove = DESTMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Two faces, opposite to origin and destination, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = ORGMOVE;
						}
						else {
							nextmove = DESTMOVE;
						}
					}
				}
				else {
					if (oriapex < 0) {
						// Two faces, opposite to origin and apex, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = ORGMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Only the face opposite to origin is viable.
						nextmove = ORGMOVE;
					}
				}
			}
			else {
				if (oridest < 0) {
					if (oriapex < 0) {
						// Two faces, opposite to destination and apex, are viable.
						//s = randomnation(2); // 's' is in {0,1}.
						if (cudamesh_randomnation(&randomseed, 2)) {
							nextmove = DESTMOVE;
						}
						else {
							nextmove = APEXMOVE;
						}
					}
					else {
						// Only the face opposite to destination is viable.
						nextmove = DESTMOVE;
					}
				}
				else {
					if (oriapex < 0) {
						// Only the face opposite to apex is viable.
						nextmove = APEXMOVE;
					}
					else {
						// The point we seek must be on the boundary of or inside this
						//   tetrahedron. Check for boundary cases.
						if (oriorg == 0) {
							// Go to the face opposite to origin.
							cudamesh_enextesymself(*searchtet);
							if (oridest == 0) {
								cudamesh_eprevself(*searchtet); // edge oppo->apex
								if (oriapex == 0) {
									// oppo is duplicated with p.
									loc = ONVERTEX; // return ONVERTEX;
									break;
								}
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							if (oriapex == 0) {
								cudamesh_enextself(*searchtet); // edge dest->oppo
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							loc = ONFACE; // return ONFACE;
							break;
						}
						if (oridest == 0) {
							// Go to the face opposite to destination.
							cudamesh_eprevesymself(*searchtet);
							if (oriapex == 0) {
								cudamesh_eprevself(*searchtet); // edge oppo->org
								loc = ONEDGE; // return ONEDGE;
								break;
							}
							loc = ONFACE; // return ONFACE;
							break;
						}
						if (oriapex == 0) {
							// Go to the face opposite to apex
							cudamesh_esymself(*searchtet);
							loc = ONFACE; // return ONFACE;
							break;
						}
						loc = INTETRAHEDRON; // return INTETRAHEDRON;
						break;
					}
				}
			}

			// Move to the selected face.
			if (nextmove == ORGMOVE) {
				cudamesh_enextesymself(*searchtet);
			}
			else if (nextmove == DESTMOVE) {
				cudamesh_eprevesymself(*searchtet);
			}
			else {
				cudamesh_esymself(*searchtet);
			}
			// Move to the adjacent tetrahedron (maybe a hull tetrahedron).
			cudamesh_fsymself(*searchtet, d_neighborlist);
			if (cudamesh_oppo(*searchtet, d_tetlist) == -1) {
				loc = OUTSIDE; // return OUTSIDE;
				break;
			}

			// Retreat the three vertices of the base face.
			torg = cudamesh_id2pointlist(cudamesh_org(*searchtet, d_tetlist), d_pointlist);
			tdest = cudamesh_id2pointlist(cudamesh_dest(*searchtet, d_tetlist), d_pointlist);
			tapex = cudamesh_id2pointlist(cudamesh_apex(*searchtet, d_tetlist), d_pointlist);

			step++;
		} // while (true)

		d_pointlocation[threadId] = loc;

		if (loc == ONVERTEX)
		{
			d_tetstatus[tetid].setAbortive(true);
			d_threadmarker[threadId] = -1;
		}
	}
}

// Split encroached segment
__device__ int checkseg4split(
	trihandle *chkseg, 
	int& encpt, 
	REAL* pointlist, 
	int* seglist,
	tethandle* seg2tetlist,
	int* tetlist,
	tethandle* neighborlist
)
{
	REAL ccent[3], len, r;
	int i;

	REAL* forg = cudamesh_id2pointlist(cudamesh_sorg(*chkseg, seglist), pointlist);
	REAL* fdest = cudamesh_id2pointlist(cudamesh_sdest(*chkseg, seglist), pointlist);

	// Initialize the return values.
	encpt = -1;

	len = cudamesh_distance(forg, fdest);
	r = 0.5 * len;
	for (i = 0; i < 3; i++) {
		ccent[i] = 0.5 * (forg[i] + fdest[i]);
	}

	// Check if it is encroached.
	// Comment: There may exist more than one encroaching points of this segment. 
	//   The 'encpt' returns the one which is closet to it.
	tethandle searchtet, spintet;
	int eapex;
	REAL d, diff, smdist = 0;
	int t1ver;

	cudamesh_sstpivot1(*chkseg, searchtet, seg2tetlist);
	spintet = searchtet;
	while (1) {
		eapex = cudamesh_apex(spintet, tetlist);
		if (eapex != -1) {
			d = cudamesh_distance(ccent, cudamesh_id2pointlist(eapex, pointlist));
			diff = d - r;
			if (fabs(diff) / r < EPSILON) diff = 0.0; // Rounding.
			if (diff < 0) {
				// This segment is encroached by eapex.
				if (encpt == -1) {
					encpt = eapex;
					smdist = d;
				}
				else {
					// Choose the closet encroaching point.
					if (d < smdist) {
						encpt = eapex;
						smdist = d;
					}
				}
			}
		}
		cudamesh_fnextself(spintet, neighborlist);
		if (spintet.id == searchtet.id) break;
	} // while (1)

	if (encpt != -1) {
		return 1;
	}

	return 0; // No need to split it.
}

__device__ int checkseg4encroach(
	REAL *pa, REAL* pb, REAL* checkpt
)
{
	// Check if the point lies inside the diametrical sphere of this seg. 
	REAL v1[3], v2[3];

	v1[0] = pa[0] - checkpt[0];
	v1[1] = pa[1] - checkpt[1];
	v1[2] = pa[2] - checkpt[2];
	v2[0] = pb[0] - checkpt[0];
	v2[1] = pb[1] - checkpt[1];
	v2[2] = pb[2] - checkpt[2];

	if (cudamesh_dot(v1, v2) < 0)
		return 1;

	return 0;
}

__global__ void kernelMarkAllEncsegs(
	REAL * d_pointlist,
	int* d_seglist,
	tethandle* d_seg2tetlist,
	int* d_segencmarker,
	int* d_tetlist,
	tethandle* d_neighborlist,
	int numofsubseg
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofsubseg)
		return;

	trihandle chkseg(pos, 0);
	int encpt;

	checkseg4split(
		&chkseg, encpt, 
		d_pointlist, d_seglist, d_seg2tetlist, d_tetlist, d_neighborlist);

	d_segencmarker[pos] = encpt;
}

__device__ void projectpoint2edge(
	REAL* p,
	REAL* e1,
	REAL* e2,
	REAL* prj
)
{
	REAL v1[3], v2[3];
	REAL len, l_p;

	v1[0] = e2[0] - e1[0];
	v1[1] = e2[1] - e1[1];
	v1[2] = e2[2] - e1[2];
	v2[0] = p[0] - e1[0];
	v2[1] = p[1] - e1[1];
	v2[2] = p[2] - e1[2];

	len = sqrt(cudamesh_dot(v1, v1));
	assert(len != 0.0);
	v1[0] /= len;
	v1[1] /= len;
	v1[2] /= len;
	l_p = cudamesh_dot(v1, v2);

	prj[0] = e1[0] + l_p * v1[0];
	prj[1] = e1[1] + l_p * v1[1];
	prj[2] = e1[2] + l_p * v1[2];
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int encsegidx = d_encseglist[threadId];

	trihandle seg(encsegidx, 0);
	REAL* ei = cudamesh_id2pointlist(cudamesh_sorg(seg, d_seglist), d_pointlist);
	REAL* ej = cudamesh_id2pointlist(cudamesh_sdest(seg, d_seglist), d_pointlist);
	int adjflag = 0, i;

	REAL* steinpt = cudamesh_id2pointlist(threadId, d_steinerptlist);
	int refptidx = d_segencmarker[encsegidx];
	assert(refptidx >= 0);
	if (refptidx != MAXINT)
	{
		REAL* refpt = cudamesh_id2pointlist(refptidx, d_pointlist);
		REAL L, L1, t;

		if (d_pointtypelist[refptidx] == FREESEGVERTEX) {
			trihandle parentseg;
			parentseg = d_point2trilist[refptidx];
			int sidx1 = d_seg2parentlist[parentseg.id];
			int idx_pi = d_segparentlist[sidx1 * 2];
			int idx_pj = d_segparentlist[sidx1 * 2 + 1];
			REAL* far_pi = cudamesh_id2pointlist(idx_pi, d_pointlist);
			REAL* far_pj = cudamesh_id2pointlist(idx_pj, d_pointlist);

			int sidx2 = d_seg2parentlist[seg.id];
			int idx_ei = d_segparentlist[sidx2 * 2];
			int idx_ej = d_segparentlist[sidx2 * 2 + 1];
			REAL* far_ei = cudamesh_id2pointlist(idx_ei, d_pointlist);
			REAL* far_ej = cudamesh_id2pointlist(idx_ej, d_pointlist);

			if ((idx_pi == idx_ei) || (idx_pj == idx_ei)) {
				// Create a Steiner point at the intersection of the segment
				//   [far_ei, far_ej] and the sphere centered at far_ei with 
				//   radius |far_ei - refpt|.
				L = cudamesh_distance(far_ei, far_ej);
				L1 = cudamesh_distance(far_ei, refpt);
				t = L1 / L;
				for (i = 0; i < 3; i++) {
					steinpt[i] = far_ei[i] + t * (far_ej[i] - far_ei[i]);
				}
				adjflag = 1;
			}
			else if ((idx_pi == idx_ej) || (idx_pj == idx_ej)) {
				L = cudamesh_distance(far_ei, far_ej);
				L1 = cudamesh_distance(far_ej, refpt);
				t = L1 / L;
				for (i = 0; i < 3; i++) {
					steinpt[i] = far_ej[i] + t * (far_ei[i] - far_ej[i]);
				}
				adjflag = 1;
			}
			else {
				// Cut the segment by the projection point of refpt.
				projectpoint2edge(refpt, ei, ej, steinpt);
			}
		}
		else {
			// Cut the segment by the projection point of refpt.
			projectpoint2edge(refpt, ei, ej, steinpt);
		}

		// Make sure that steinpt is not too close to ei and ej.
		L = cudamesh_distance(ei, ej);
		L1 = cudamesh_distance(steinpt, ei);
		t = L1 / L;
		if ((t < 0.2) || (t > 0.8)) {
			// Split the point at the middle.
			for (i = 0; i < 3; i++) {
				steinpt[i] = ei[i] + 0.5 * (ej[i] - ei[i]);
			}
		}
	}
	else
	{
		// Split the point at the middle.
		for (i = 0; i < 3; i++) {
			steinpt[i] = ei[i] + 0.5 * (ej[i] - ei[i]);
		}
	}
}

// Split encroached subface
__device__ int checkface4split(
	trihandle *chkfac,
	int& encpt,
	REAL* pointlist,
	int* trifacelist,
	tethandle* tri2tetlist,
	int* tetlist
)
{
	REAL *pa, *pb, *pc;
	REAL area, rd, len;
	REAL A[4][4], rhs[4], cent[3], D;
	int indx[4];
	int i;

	encpt = -1;

	pa = cudamesh_id2pointlist(cudamesh_sorg(*chkfac, trifacelist), pointlist);
	pb = cudamesh_id2pointlist(cudamesh_sdest(*chkfac, trifacelist), pointlist);
	pc = cudamesh_id2pointlist(cudamesh_sapex(*chkfac, trifacelist), pointlist);

	// Compute the coefficient matrix A (3x3).
	A[0][0] = pb[0] - pa[0];
	A[0][1] = pb[1] - pa[1];
	A[0][2] = pb[2] - pa[2]; // vector V1 (pa->pb)
	A[1][0] = pc[0] - pa[0];
	A[1][1] = pc[1] - pa[1];
	A[1][2] = pc[2] - pa[2]; // vector V2 (pa->pc)
	cudamesh_cross(A[0], A[1], A[2]); // vector V3 (V1 X V2)

	area = 0.5 * sqrt(cudamesh_dot(A[2], A[2])); // The area of [a,b,c].

										// Compute the right hand side vector b (3x1).
	rhs[0] = 0.5 * cudamesh_dot(A[0], A[0]); // edge [a,b]
	rhs[1] = 0.5 * cudamesh_dot(A[1], A[1]); // edge [a,c]
	rhs[2] = 0.0;

	// Solve the 3 by 3 equations use LU decomposition with partial 
	//   pivoting and backward and forward substitute.
	if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
		// A degenerate triangle. 
		//printf("checkface4split: A degenerate subface!\n");
		encpt = -1;
		return -1;
	}

	cudamesh_lu_solve(A, 3, indx, rhs, 0);
	cent[0] = pa[0] + rhs[0];
	cent[1] = pa[1] + rhs[1];
	cent[2] = pa[2] + rhs[2];
	rd = sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);

	tethandle searchtet;
	REAL smlen = 0;

	// Check if this subface is locally encroached.
	for (i = 0; i < 2; i++) {
		cudamesh_stpivot(*chkfac, searchtet, tri2tetlist);
		if (!cudamesh_ishulltet(searchtet, tetlist)) {
			len = cudamesh_distance(
				cudamesh_id2pointlist(cudamesh_oppo(searchtet, tetlist), pointlist), 
				cent);
			if ((fabs(len - rd) / rd) < EPSILON) len = rd;// Rounding.
			if (len < rd) {
				if (smlen == 0) {
					smlen = len;
					encpt = cudamesh_oppo(searchtet, tetlist);
				}
				else {
					if (len < smlen) {
						smlen = len;
						encpt = cudamesh_oppo(searchtet, tetlist);
					}
				}
			}
		}
		cudamesh_sesymself(*chkfac);
	}

	return encpt != -1;
}

__device__ int checkface4encroach(
	REAL *pa, REAL *pb, REAL *pc, REAL *checkpt
)
{
	REAL rd, len, cent[3];

	cudamesh_circumsphere(pa, pb, pc, NULL, cent, &rd);
	assert(rd != 0);
	len = cudamesh_distance(cent, checkpt);
	if ((fabs(len - rd) / rd) < EPSILON) len = rd; // Rounding.

	if (len < rd) {
		// The point lies inside the circumsphere of this face.
		return 1;  // Encroached.
	}

	return 0;
}

__global__ void kernelMarkAllEncsubfaces(
	REAL * d_pointlist,
	int* d_trifacelist,
	tethandle* d_tri2tetlist,
	int* d_subfaceencmarker,
	int* d_tetlist,
	int numofsubface
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofsubface)
		return;

	trihandle chkfac(pos, 0);
	int encpt;

	checkface4split(
		&chkfac, encpt,
		d_pointlist, d_trifacelist, d_tri2tetlist, d_tetlist);

	d_subfaceencmarker[pos] = encpt;
}

__global__ void kernelComputeSteinerPoint_Subface(
	REAL* d_pointlist,
	int* d_trifacelist,
	tristatus* d_tristatus,
	int* d_encsubfacelist,
	REAL* d_steinerptlist,
	int numofencsubface
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofencsubface)
		return;

	int encsubfaceidx = d_encsubfacelist[pos];

	REAL *pa, *pb, *pc;
	REAL area, rd, len;
	REAL A[4][4], rhs[4], D;
	int indx[4];
	int i;

	trihandle chkfac(encsubfaceidx, 0);
	REAL* steinpt = cudamesh_id2pointlist(pos, d_steinerptlist);

	pa = cudamesh_id2pointlist(cudamesh_sorg(chkfac, d_trifacelist), d_pointlist);
	pb = cudamesh_id2pointlist(cudamesh_sdest(chkfac, d_trifacelist), d_pointlist);
	pc = cudamesh_id2pointlist(cudamesh_sapex(chkfac, d_trifacelist), d_pointlist);

	// Compute the coefficient matrix A (3x3).
	A[0][0] = pb[0] - pa[0];
	A[0][1] = pb[1] - pa[1];
	A[0][2] = pb[2] - pa[2]; // vector V1 (pa->pb)
	A[1][0] = pc[0] - pa[0];
	A[1][1] = pc[1] - pa[1];
	A[1][2] = pc[2] - pa[2]; // vector V2 (pa->pc)
	cudamesh_cross(A[0], A[1], A[2]); // vector V3 (V1 X V2)

	area = 0.5 * sqrt(cudamesh_dot(A[2], A[2])); // The area of [a,b,c].

												 // Compute the right hand side vector b (3x1).
	rhs[0] = 0.5 * cudamesh_dot(A[0], A[0]); // edge [a,b]
	rhs[1] = 0.5 * cudamesh_dot(A[1], A[1]); // edge [a,c]
	rhs[2] = 0.0;

	// Solve the 3 by 3 equations use LU decomposition with partial 
	//   pivoting and backward and forward substitute.
	if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
		// A degenerate triangle. 
		printf("kernelComputeSteinerPointOnSubface: A degenerate subface. This should not happen!\n");
	}

	cudamesh_lu_solve(A, 3, indx, rhs, 0);
	steinpt[0] = pa[0] + rhs[0];
	steinpt[1] = pa[1] + rhs[1];
	steinpt[2] = pa[2] + rhs[2];
}

// Split bad tets
__device__ int checktet4split(
	tethandle* chktet,
	REAL* pointlist,
	int* tetlist,
	REAL minratio
)
{
	int ipa, ipb, ipc, ipd;
	REAL *pa, *pb, *pc, *pd;
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];
	REAL smlen = 0, rd;
	REAL A[4][4], rhs[4], D;
	int indx[4];
	int i;

	ipd = tetlist[4*(*chktet).id + 3];
	if (ipd == -1) {
		return 0; // Do not split a hull tet.
	}

	ipa = tetlist[4*(*chktet).id + 0];
	ipb = tetlist[4*(*chktet).id + 1];
	ipc = tetlist[4*(*chktet).id + 2];

	pa = cudamesh_id2pointlist(ipa, pointlist);
	pb = cudamesh_id2pointlist(ipb, pointlist);
	pc = cudamesh_id2pointlist(ipc, pointlist);
	pd = cudamesh_id2pointlist(ipd, pointlist);

	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	// Set the matrix A = [vda, vdb, vdc]^T.
	for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
		// A degenerated tet (vol = 0).
		// This is possible due to the use of exact arithmetic.  We temporarily
		//   leave this tet. It should be fixed by mesh optimization.
		return 0;
	}

	  // Check the radius-edge ratio. Set by -q#.
	if (minratio > 0) {
		// Calculate the circumcenter and radius of this tet.
		rhs[0] = 0.5 * cudamesh_dot(vda, vda);
		rhs[1] = 0.5 * cudamesh_dot(vdb, vdb);
		rhs[2] = 0.5 * cudamesh_dot(vdc, vdc);
		cudamesh_lu_solve(A, 3, indx, rhs, 0);
		rd = sqrt(cudamesh_dot(rhs, rhs));
		// Calculate the shortest edge length.
		elen[0] = cudamesh_dot(vda, vda);
		elen[1] = cudamesh_dot(vdb, vdb);
		elen[2] = cudamesh_dot(vdc, vdc);
		elen[3] = cudamesh_dot(vab, vab);
		elen[4] = cudamesh_dot(vbc, vbc);
		elen[5] = cudamesh_dot(vca, vca);
		smlen = elen[0]; //sidx = 0;
		for (i = 1; i < 6; i++) {
			if (smlen > elen[i]) {
				smlen = elen[i]; //sidx = i; 
			}
		}
		smlen = sqrt(smlen);
		D = rd / smlen;
		if (D > minratio) {
			// A bad radius-edge ratio.
			return 1;
		}
	}

	return 0;
}

__global__ void kernelMarkAllBadtets(
	REAL* d_pointlist,
	int* d_tetlist,
	tetstatus* d_tetstatus,
	REAL minratio,
	int numofbadtet
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofbadtet)
		return;

	tethandle chktet(pos, 11);
	if (checktet4split(&chktet, d_pointlist, d_tetlist, minratio))
	{
		d_tetstatus[pos].setBad(true);
	}
}

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
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	int threadId = d_threadlist[pos];
	int tetid = d_tetidlist[threadId];
	REAL* steinpt = d_insertptlist + 3 * threadId;

	int ipa, ipb, ipc, ipd;
	REAL *pa, *pb, *pc, *pd;
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];
	REAL smlen = 0, rd;
	REAL A[4][4], rhs[4], D;
	int indx[4];
	int i;

	ipd = d_tetlist[4 * tetid + 3];
	if (ipd == -1) {
		// This should not happend
		printf("Thread #%d - Error: Try to split a hull tet #%d!\n", threadId, tetid);
		return;
	}

	ipa = d_tetlist[4 * tetid + 0];
	ipb = d_tetlist[4 * tetid + 1];
	ipc = d_tetlist[4 * tetid + 2];

	pa = cudamesh_id2pointlist(ipa, d_pointlist);
	pb = cudamesh_id2pointlist(ipb, d_pointlist);
	pc = cudamesh_id2pointlist(ipc, d_pointlist);
	pd = cudamesh_id2pointlist(ipd, d_pointlist);

	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	// Set the matrix A = [vda, vdb, vdc]^T.
	for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	//if (cuda_orient3d(pa, pb, pc, pd) < 0.001 && cuda_orient3d(pa, pb, pc, pd) > -0.001)
	//{
		// Nearly degenerated tet.
		// Set to abortive to avoid invalid point coordinate
		//d_tetstatus[tetid].setAbortive(true);
		//d_threadmarker[threadId] = -1;
		//return;
	//}

	if (!cudamesh_lu_decmp(A, 3, indx, &D, 0)) {
		// This should not happend
		//printf("Thread #%d - Error: Try to split a degenerated tet #%d!\n", threadId, tetid);
		d_tetstatus[tetid].setAbortive(true);
		d_threadmarker[threadId] = -1;
		return;
	}

	// Calculate the circumcenter and radius of this tet.
	rhs[0] = 0.5 * cudamesh_dot(vda, vda);
	rhs[1] = 0.5 * cudamesh_dot(vdb, vdb);
	rhs[2] = 0.5 * cudamesh_dot(vdc, vdc);
	cudamesh_lu_solve(A, 3, indx, rhs, 0);
	for (i = 0; i < 3; i++)
	{
		steinpt[i] = pd[i] + rhs[i];
	}

	// set priority
	//rd = sqrt(cudamesh_dot(rhs, rhs));
	//Calculate the shortest edge length.
	elen[0] = cudamesh_dot(vda, vda);
	elen[1] = cudamesh_dot(vdb, vdb);
	elen[2] = cudamesh_dot(vdc, vdc);
	elen[3] = cudamesh_dot(vab, vab);
	elen[4] = cudamesh_dot(vbc, vbc);
	elen[5] = cudamesh_dot(vca, vca);
	//Use radius-to-shortest-edge radio as priority
	//smlen = elen[0]; //sidx = 0;
	//for (i = 1; i < 6; i++) {
	//	if (smlen > elen[i]) {
	//		smlen = elen[i]; //sidx = i; 
	//	}
	//}
	//smlen = sqrt(smlen);
	//d_priority[threadId] = __float_as_int((float)(smlen / rd));

	// Use volume as priority
	// Use heron-type formula to compute the volume of a tetrahedron
	// https://en.wikipedia.org/wiki/Heron%27s_formula
	//if (cuda_orient3d(pa, pb, pc, pd) < 0.001 && cuda_orient3d(pa, pb, pc, pd) > -0.001)
	//{
	//	d_priority[threadId] = MAXINT;
	//}
	//else
	{
		REAL U, V, W, u, v, w; // first three form a triangle; u opposite to U and so on
		REAL X, x, Y, y, Z, z;
		REAL a, b, c, d;
		U = sqrt(elen[3]); //ab
		V = sqrt(elen[4]); //bc
		W = sqrt(elen[5]); //ca
		u = sqrt(elen[2]); //dc
		v = sqrt(elen[0]); //da
		w = sqrt(elen[1]); //db

		X = (w - U + v)*(U + v + w);
		x = (U - v + w)*(v - w + U);
		Y = (u - V + w)*(V + w + u);
		y = (V - w + u)*(w - u + V);
		Z = (v - W + u)*(W + u + v);
		z = (W - u + v)*(u - v + W);

		a = sqrt(x*Y*Z);
		b = sqrt(y*Z*X);
		c = sqrt(z*X*Y);
		d = sqrt(x*y*z);

		REAL vol = sqrt((-a + b + c + d)*(a - b + c + d)*(a + b - c + d)*(a + b + c - d)) / (192 * u*v*w);
		d_priority[threadId] = __float_as_int((float)(1 / vol));
		//d_priority[threadId] = __float_as_int((float)(1 / rd));
	}

	//if (cuda_orient3d(pa, pb, pc, pd) < 0.001 && cuda_orient3d(pa, pb, pc, pd) > -0.001)
	//{
	//	if(pos < 100)
	//		printf("%d ", d_priority[threadId]);
	//}
	//if (pos < 100)
	//	printf("Tet #%d: (%lf, %lf, %lf), (%lf, %lf, %lf), (%lf, %lf, %lf), (%lf, %lf, %lf) | (%lf, %lf, %lf) | %lf\n", 
	//		tetid, 
	//		pa[0], pa[1], pa[2],
	//		pb[0], pb[1], pb[2],
	//		pc[0], pc[1], pc[2],
	//		pd[0], pd[1], pd[2],
	//		steinpt[0], steinpt[1], steinpt[2],
	//		cuda_orient3d(pa, pb, pc, pd));
	//if (pos < 100)
	//	printf("%d ", d_priority[threadId]);
}

__global__ void kernelCompactSeg(
	int* d_seglist,
	int* d_sizes,
	int* d_indices,
	int* d_list,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_sizes[pos] == 0)
		return;

	int index = d_indices[pos];
	d_list[2 * index + 0] = d_seglist[3 * pos + 0];
	d_list[2 * index + 1] = d_seglist[3 * pos + 1];
}

__global__ void kernelCompactTriface(
	int* d_trifacelist,
	int* d_sizes,
	int* d_indices,
	int* d_list,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_sizes[pos] == 0)
		return;

	int index = d_indices[pos];
	d_list[3 * index + 0] = d_trifacelist[3 * pos + 0];
	d_list[3 * index + 1] = d_trifacelist[3 * pos + 1];
	d_list[3 * index + 2] = d_trifacelist[3 * pos + 2];
}

__global__ void kernelCompactTet_Phase1(
	int* d_tetlist,
	tetstatus* d_tetstatus,
	int* d_sizes,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_tetstatus[pos].isEmpty())
		d_sizes[pos] = 0;

	if (d_tetlist[4 * pos + 3] == -1)
		d_sizes[pos] = 0;
}

__global__ void kernelCompactTet_Phase2(
	int* d_tetlist,
	int* d_sizes,
	int* d_indices,
	int* d_list,
	int numofthreads
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numofthreads)
		return;

	if (d_sizes[pos] == 0)
		return;

	int index = d_indices[pos];
	d_list[4 * index + 0] = d_tetlist[4 * pos + 0];
	d_list[4 * index + 1] = d_tetlist[4 * pos + 1];
	d_list[4 * index + 2] = d_tetlist[4 * pos + 2];
	d_list[4 * index + 3] = d_tetlist[4 * pos + 3];
}