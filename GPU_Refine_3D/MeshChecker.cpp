#include "Mesh.h"
#include "MeshPredicates.h"
#include "MeshReconstruct.h"
#include "MeshChecker.h"
#include <stdio.h>
#include <math.h>

// Split bad tets
int isBadTet(
	REAL *pa, REAL *pb, REAL *pc, REAL *pd,
	REAL minratio
)
{
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];
	REAL smlen = 0, rd;
	REAL A[4][4], rhs[4], D;
	int indx[4];
	int i;

	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	// Set the matrix A = [vda, vdb, vdc]^T.
	for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	//REAL ret = orient3d(pa, pb, pc, pd);
	//if (ret > -0.001 && ret < 0.001)
	//{
	//	return 1;
	//}

	if (!meshludecmp(A, 3, indx, &D, 0)) {
		// A degenerated tet (vol = 0).
		// This is possible due to the use of exact arithmetic.  We temporarily
		//   leave this tet. It should be fixed by mesh optimization.
		return 0;
	}

	// Check the radius-edge ratio. Set by -q#.
	if (minratio > 0) {
		// Calculate the circumcenter and radius of this tet.
		rhs[0] = 0.5 * meshdot(vda, vda);
		rhs[1] = 0.5 * meshdot(vdb, vdb);
		rhs[2] = 0.5 * meshdot(vdc, vdc);
		meshlusolve(A, 3, indx, rhs, 0);
		rd = sqrt(meshdot(rhs, rhs));
		// Calculate the shortest edge length.
		elen[0] = meshdot(vda, vda);
		elen[1] = meshdot(vdb, vdb);
		elen[2] = meshdot(vdc, vdc);
		elen[3] = meshdot(vab, vab);
		elen[4] = meshdot(vbc, vbc);
		elen[5] = meshdot(vca, vca);
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

int isBadTet(
	REAL *pa, REAL *pb, REAL *pc, REAL *pd,
	REAL minratio, REAL* ccent
)
{
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];
	REAL smlen = 0, rd;
	REAL A[4][4], rhs[4], D;
	int indx[4];
	int i;

	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	// Set the matrix A = [vda, vdb, vdc]^T.
	for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	//REAL ret = orient3d(pa, pb, pc, pd);
	//if (ret > -0.001 && ret < 0.001)
	//{
	//	return 0;
	//}

	if (!meshludecmp(A, 3, indx, &D, 0)) {
		// A degenerated tet (vol = 0).
		// This is possible due to the use of exact arithmetic.  We temporarily
		//   leave this tet. It should be fixed by mesh optimization.
		return 0;
	}

	// Check the radius-edge ratio. Set by -q#.
	if (minratio > 0) {
		// Calculate the circumcenter and radius of this tet.
		rhs[0] = 0.5 * meshdot(vda, vda);
		rhs[1] = 0.5 * meshdot(vdb, vdb);
		rhs[2] = 0.5 * meshdot(vdc, vdc);
		meshlusolve(A, 3, indx, rhs, 0);
		for (i = 0; i < 3; i++) ccent[i] = pd[i] + rhs[i];
		rd = sqrt(meshdot(rhs, rhs));
		// Calculate the shortest edge length.
		elen[0] = meshdot(vda, vda);
		elen[1] = meshdot(vdb, vdb);
		elen[2] = meshdot(vdc, vdc);
		elen[3] = meshdot(vab, vab);
		elen[4] = meshdot(vbc, vbc);
		elen[5] = meshdot(vca, vca);
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

REAL calRatio(
	REAL *pa, REAL *pb, REAL *pc, REAL *pd
)
{
	REAL vda[3], vdb[3], vdc[3];
	REAL vab[3], vbc[3], vca[3];
	REAL elen[6];
	REAL smlen = 0, rd;
	REAL A[4][4], rhs[4], ccent[3], D;
	int indx[4];
	int i;

	// Get the edge vectors vda: d->a, vdb: d->b, vdc: d->c.
	// Set the matrix A = [vda, vdb, vdc]^T.
	for (i = 0; i < 3; i++) A[0][i] = vda[i] = pa[i] - pd[i];
	for (i = 0; i < 3; i++) A[1][i] = vdb[i] = pb[i] - pd[i];
	for (i = 0; i < 3; i++) A[2][i] = vdc[i] = pc[i] - pd[i];

	// Get the other edge vectors.
	for (i = 0; i < 3; i++) vab[i] = pb[i] - pa[i];
	for (i = 0; i < 3; i++) vbc[i] = pc[i] - pb[i];
	for (i = 0; i < 3; i++) vca[i] = pa[i] - pc[i];

	if (!meshludecmp(A, 3, indx, &D, 0)) {
		// A degenerated tet (vol = 0).
		// This is possible due to the use of exact arithmetic.  We temporarily
		//   leave this tet. It should be fixed by mesh optimization.
		return -1.0;
	}

	// Calculate the circumcenter and radius of this tet.
	rhs[0] = 0.5 * meshdot(vda, vda);
	rhs[1] = 0.5 * meshdot(vdb, vdb);
	rhs[2] = 0.5 * meshdot(vdc, vdc);
	meshlusolve(A, 3, indx, rhs, 0);
	for (i = 0; i < 3; i++) ccent[i] = pd[i] + rhs[i];
	rd = sqrt(meshdot(rhs, rhs));
	// Calculate the shortest edge length.
	elen[0] = meshdot(vda, vda);
	elen[1] = meshdot(vdb, vdb);
	elen[2] = meshdot(vdc, vdc);
	elen[3] = meshdot(vab, vab);
	elen[4] = meshdot(vbc, vbc);
	elen[5] = meshdot(vca, vca);
	smlen = elen[0]; //sidx = 0;
	for (i = 1; i < 6; i++) {
		if (smlen > elen[i]) {
			smlen = elen[i]; //sidx = i; 
		}
	}
	smlen = sqrt(smlen);
	D = rd / smlen;

	return D;
}

void calDihedral(
	REAL **p,
	REAL *alldihed
)
{
	REAL A[4][4], rhs[4], D;
	REAL V[6][3], N[4][3], H[4]; // edge-vectors, face-normals, face-heights.
	int indx[4];

	int i, j;
	// Set the edge vectors: V[0], ..., V[5]
	for (i = 0; i < 3; i++) V[0][i] = p[0][i] - p[3][i]; // V[0]: p3->p0.
	for (i = 0; i < 3; i++) V[1][i] = p[1][i] - p[3][i]; // V[1]: p3->p1.
	for (i = 0; i < 3; i++) V[2][i] = p[2][i] - p[3][i]; // V[2]: p3->p2.
	for (i = 0; i < 3; i++) V[3][i] = p[1][i] - p[0][i]; // V[3]: p0->p1.
	for (i = 0; i < 3; i++) V[4][i] = p[2][i] - p[1][i]; // V[4]: p1->p2.
	for (i = 0; i < 3; i++) V[5][i] = p[0][i] - p[2][i]; // V[5]: p2->p0.

	// Set the matrix A = [V[0], V[1], V[2]]^T.
	for (j = 0; j < 3; j++) {
		for (i = 0; i < 3; i++) A[j][i] = V[j][i];
	}

	// Decompose A just once.
	if (meshludecmp(A, 3, indx, &D, 0)) {
		// Get the three faces normals.
		for (j = 0; j < 3; j++) {
			for (i = 0; i < 3; i++) rhs[i] = 0.0;
			rhs[j] = 1.0;  // Positive means the inside direction
			meshlusolve(A, 3, indx, rhs, 0);
			for (i = 0; i < 3; i++) N[j][i] = rhs[i];
		}
		// Get the fourth face normal by summing up the first three.
		for (i = 0; i < 3; i++) N[3][i] = -N[0][i] - N[1][i] - N[2][i];
		// Get the radius of the circumsphere.
		for (i = 0; i < 3; i++) rhs[i] = 0.5 * meshdot(V[i], V[i]);
		meshlusolve(A, 3, indx, rhs, 0);
		// Normalize the face normals.
		for (i = 0; i < 4; i++) {
			// H[i] is the inverse of height of its corresponding face.
			H[i] = sqrt(meshdot(N[i], N[i]));
			for (j = 0; j < 3; j++) N[i][j] /= H[i];
		}
	}
	else {
		// Calculate the four face normals.
		meshfacenormal(p[2], p[1], p[3], N[0], 1, NULL);
		meshfacenormal(p[0], p[2], p[3], N[1], 1, NULL);
		meshfacenormal(p[1], p[0], p[3], N[2], 1, NULL);
		meshfacenormal(p[0], p[1], p[2], N[3], 1, NULL);
		// Normalize the face normals.
		for (i = 0; i < 4; i++) {
			// H[i] is the twice of the area of the face.
			H[i] = sqrt(meshdot(N[i], N[i]));
			for (j = 0; j < 3; j++) N[i][j] /= H[i];
		}
	}

	// Get the dihedrals (in degree) at each edges.
	j = 0;
	for (i = 1; i < 4; i++) {
		alldihed[j] = -meshdot(N[0], N[i]); // Edge cd, bd, bc.
		if (alldihed[j] < -1.0) alldihed[j] = -1; // Rounding.
		else if (alldihed[j] > 1.0) alldihed[j] = 1;
		alldihed[j] = acos(alldihed[j]) / PI * 180.0;
		j++;
	}
	for (i = 2; i < 4; i++) {
		alldihed[j] = -meshdot(N[1], N[i]); // Edge ad, ac.
		if (alldihed[j] < -1.0) alldihed[j] = -1; // Rounding.
		else if (alldihed[j] > 1.0) alldihed[j] = 1;
		alldihed[j] = acos(alldihed[j]) / PI * 180.0;
		j++;
	}
	alldihed[j] = -meshdot(N[2], N[3]); // Edge ab.
	if (alldihed[j] < -1.0) alldihed[j] = -1; // Rounding.
	else if (alldihed[j] > 1.0) alldihed[j] = 1;
	alldihed[j] = acos(alldihed[j]) / PI * 180.0;
}

int countBadTets(
	double* pointlist,
	int* tetlist,
	int numoftet,
	double minratio
)
{
	int ip[4];
	REAL* p[4];
	int count = 0;
	int i, j;
	for (i = 0; i < numoftet; i++)
	{
		for (j = 0; j < 4; j++)
		{
			ip[j] = tetlist[4 * i + j];
			p[j] = id2pointlist(ip[j], pointlist);
		}
		if (isBadTet(p[0], p[1], p[2], p[3], minratio))
			count++;
	}

	return count;
}

void checkCDTMesh(
	int numofpoint,
	double* pointlist,
	int numofedge,
	int* edgelist,
	int numoftriface,
	int* trifacelist,
	int numoftet,
	int* tetlist
)
{
	printf("*****************************************\n");
	printf("Checking CDT Mesh......\n");
	printf("-----------------------------------------\n");
	printf("Mesh Size:\n");
	printf("	Number of point = %d\n", numofpoint);
	printf("	Number of edge = %d\n", numofedge);
	printf("	Number of triface = %d\n", numoftriface);
	printf("	Number of tetrahedron = %d\n", numoftet);
	printf("-----------------------------------------\n");
	printf("Reconstruct mesh...\n");

	// output variables
	tethandle* outpoint2tetlist;
	trihandle* outpoint2trilist;
	verttype* outpointtypelist;
	int outnumofedge;
	int* outseglist;
	trihandle* outseg2trilist;
	tethandle* outseg2tetlist;
	int outnumoftriface;
	int* outtrifacelist;
	tethandle* outtri2tetlist;
	trihandle* outtri2trilist;
	trihandle* outtri2seglist;
	int outnumoftetrahedron;
	int* outtetlist;
	tethandle* outneighborlist;
	trihandle* outtet2trilist;
	trihandle* outtet2seglist;

	MESHIO input_mesh;
	input_mesh.numofpoints = numofpoint;
	input_mesh.pointlist = pointlist;
	input_mesh.numofedges = numofedge;
	input_mesh.edgelist = edgelist;
	input_mesh.numoftrifaces = numoftriface;
	input_mesh.trifacelist = trifacelist;
	input_mesh.numoftets = numoftet;
	input_mesh.tetlist = tetlist;

	reconstructMesh(
		&input_mesh,
		outpoint2tetlist,
		outpoint2trilist,
		outpointtypelist,
		outnumofedge,
		outseglist,
		outseg2trilist,
		outseg2tetlist,
		outnumoftriface,
		outtrifacelist,
		outtri2tetlist,
		outtri2trilist,
		outtri2seglist,
		outnumoftetrahedron,
		outtetlist,
		outneighborlist,
		outtet2trilist,
		outtet2seglist,
		true
	);
	printf("-----------------------------------------\n");
	printf("Check constrained Delaunay properties...\n");

	bool result = true;
	for (int i = 0; i < outnumoftetrahedron; i++)
	{
		tethandle tmp(i, 11), neightet;
		trihandle neighsh;
		int p[5];
		double ret;
		double* cp[5];
		if (ishulltet(tmp, outtetlist))
			continue;

		// Get 4 vertices of ith tetrahedron
		p[0] = org(tmp, outtetlist);
		p[1] = dest(tmp, outtetlist);
		p[2] = apex(tmp, outtetlist);
		p[3] = oppo(tmp, outtetlist);

		for (tmp.ver = 0; tmp.ver < 4; tmp.ver++)
		{
			// Check if this face is subface
			tspivot(tmp, neighsh, outtet2trilist);
			if (neighsh.id != -1) // this face is subface, skip
				continue;

			// Get opposite vertex
			fsym(tmp, neightet, outneighborlist);
			p[4] = oppo(neightet, outtetlist);

			// Get coordinates
			for (int j = 0; j < 5; j++)
				cp[j] = id2pointlist(p[j], pointlist);

			// In-sphere test: insphere
			// Return a negative value if the orient3d() of first 4 points is negative and
			// cp[4] lies inside the sphere passing through cp[0], cp[1], cp[2] and cp[3]
			ret = insphere(cp[0], cp[1], cp[2], cp[3], cp[4]); 

			if (ret < -0.0001)
			{
				result = false;
				printf("failed: Tet #%d and Tet #%d, ret = %g\n", tmp.id, neightet.id, ret);
			}
		}
	}

	printf("Test result: ");
	if (!result)
		printf("Failed\n");
	else
		printf("Succeeded\n");

	printf("-----------------------------------------\n");
}