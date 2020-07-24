#include "Mesh.h"
#include "MeshPredicates.h"
#include "MeshReconstruct.h"
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <time.h>

// helper functions
bool tetshareface(int t1, int t2, int* tetlist)
{
	std::vector<int> list;
	for (int i = 0; i < 4; i++)
	{
		int p[2];
		p[0] = tetlist[4 * t1 + i];
		p[1] = tetlist[4 * t2 + i];
		for (int j = 0; j < 2; j++)
		{
			if (std::find(list.begin(), list.end(), p[j]) == list.end())
				list.push_back(p[j]);
		}
	}
	return (list.size() == 5);
}

bool trishareedge(int s1, int s2, int* trilist)
{
	std::vector<int> list;
	for (int i = 0; i < 3; i++)
	{
		int p[2];
		p[0] = trilist[3 * s1 + i];
		p[1] = trilist[3 * s2 + i];
		for (int j = 0; j < 2; j++)
		{
			if (std::find(list.begin(), list.end(), p[j]) == list.end())
				list.push_back(p[j]);
		}
	}
	return (list.size() == 4);
}

bool isDegenerateTet(double* pa, double *pb, double *pc, double* pd)
{
	double ret = orient3d(pa, pb, pc, pd);
	if (ret < 0.001 && ret > -0.001) // nearly degenerate
		return true;
	else
		return false;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// facenormal()    Calculate the normal of the face.                         //
//                                                                           //
// The normal of the face abc can be calculated by the cross product of 2 of //
// its 3 edge vectors.  A better choice of two edge vectors will reduce the  //
// numerical error during the calculation.  Burdakov proved that the optimal //
// basis problem is equivalent to the minimum spanning tree problem with the //
// edge length be the functional, see Burdakov, "A greedy algorithm for the  //
// optimal basis problem", BIT 37:3 (1997), 591-599. If 'pivot' > 0, the two //
// short edges in abc are chosen for the calculation.                        //
//                                                                           //
// If 'lav' is not NULL and if 'pivot' is set, the average edge length of    //
// the edges of the face [a,b,c] is returned.                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void mymeshfacenormal(REAL* pa, REAL* pb, REAL* pc, REAL *n, int pivot,
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

//////////////////////////////////////////////////////////////////////////////
//                                                                           //
// facedihedral()    Return the dihedral angle (in radian) between two       //
//                   adjoining faces.                                        //
//                                                                           //
// 'pa', 'pb' are the shared edge of these two faces, 'pc1', and 'pc2' are   //
// apexes of these two faces.  Return the angle (between 0 to 2*pi) between  //
// the normal of face (pa, pb, pc1) and normal of face (pa, pb, pc2).        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

REAL facedihedral(REAL* pa, REAL* pb, REAL* pc1, REAL* pc2)
{
	REAL n1[3], n2[3];
	REAL n1len, n2len;
	REAL costheta, ori;
	REAL theta;

	mymeshfacenormal(pa, pb, pc1, n1, 1, NULL);
	mymeshfacenormal(pa, pb, pc2, n2, 1, NULL);
	n1len = sqrt(meshdot(n1, n1));
	n2len = sqrt(meshdot(n2, n2));
	costheta = meshdot(n1, n2) / (n1len * n2len);
	// Be careful rounding error!
	if (costheta > 1.0) {
		costheta = 1.0;
	}
	else if (costheta < -1.0) {
		costheta = -1.0;
	}
	theta = acos(costheta);
	ori = orient3d(pa, pb, pc1, pc2);
	if (ori > 0.0) {
		theta = 2 * PI - theta;
	}

	return theta;
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// makepoint2submap()    Create a map from vertex to subfaces incident at it.  //
//                                                                           //
// The map is returned in two arrays 'idx2faclist' and 'facperverlist'.  All //
// subfaces incident at i-th vertex (i is counted from 0) are found in the   //
// array facperverlist[j], where idx2faclist[i] <= j < idx2faclist[i + 1].   //
// Each entry in facperverlist[j] is a subface whose origin is the vertex.   //
//                                                                           //
// NOTE: These two arrays will be created inside this routine, don't forget  //
// to free them after using.                                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void makepoint2submap(int* trilist, int*& idx2faclist,
	trihandle*& facperverlist, int numoftriface, int numofpoint)
{
	trihandle shloop;
	int i, j, k;
	//printf("  Making a map from points to subfaces.\n");

	// Initialize 'idx2faclist'.
	idx2faclist = new int[numofpoint + 1];
	for (i = 0; i < numofpoint + 1; i++) idx2faclist[i] = 0;

	// Loop all subfaces, counter the number of subfaces incident at a vertex.
	int count = 0;
	while (count < numoftriface) {
		trihandle shloop(count, 0);
		// Increment the number of incident subfaces for each vertex.
		j = trilist[3 * shloop.id];
		idx2faclist[j]++;
		j = trilist[3 * shloop.id + 1];
		idx2faclist[j]++;
		// Skip the third corner if it is a segment.
		if (trilist[3 * shloop.id + 2] != -1) {
			j = trilist[3 * shloop.id + 2];
			idx2faclist[j]++;
		}
		count++;
	}

	// Calculate the total length of array 'facperverlist'.
	j = idx2faclist[0];
	idx2faclist[0] = 0;  // Array starts from 0 element.
	for (i = 0; i < numofpoint; i++) {
		k = idx2faclist[i + 1];
		idx2faclist[i + 1] = idx2faclist[i] + j;
		j = k;
	}

	// The total length is in the last unit of idx2faclist.
	facperverlist = new trihandle[idx2faclist[i]];

	// Loop all subfaces again, remember the subfaces at each vertex.
	count = 0;
	while (count < numoftriface) {
		trihandle shloop(count, 0);
		j = trilist[3 * shloop.id];
		shloop.shver = 0; // save the origin.
		facperverlist[idx2faclist[j]] = shloop;
		idx2faclist[j]++;
		// Is it a subface or a subsegment?
		if (trilist[3 * shloop.id + 2] != -1) {
			j = trilist[3 * shloop.id + 1];
			shloop.shver = 2; // save the origin.
			facperverlist[idx2faclist[j]] = shloop;
			idx2faclist[j]++;
			j = trilist[3 * shloop.id + 2];
			shloop.shver = 4; // save the origin.
			facperverlist[idx2faclist[j]] = shloop;
			idx2faclist[j]++;
		}
		else {
			j = trilist[3 * shloop.id + 1];
			shloop.shver = 1; // save the origin.
			facperverlist[idx2faclist[j]] = shloop;
			idx2faclist[j]++;
		}
		count++;
	}

	// Contents in 'idx2faclist' are shifted, now shift them back.
	for (i = numofpoint - 1; i >= 0; i--) {
		idx2faclist[i + 1] = idx2faclist[i];
	}
	idx2faclist[0] = 0;
}

// reconstructMesh

void reconstructMesh(
	MESHIO* input_mesh,
	tethandle*& outpoint2tetlist,
	trihandle*& outpoint2trilist,
	verttype*& outpointtypelist,
	int& outnumofedge,
	int*& outseglist,
	trihandle*& outseg2trilist,
	tethandle*& outseg2tetlist,
	int& outnumoftriface,
	int*& outtrifacelist,
	tethandle*& outtri2tetlist,
	trihandle*& outtri2trilist,
	trihandle*& outtri2seglist,
	int& outnumoftetrahedron,
	int*& outtetlist,
	tethandle*& outneighborlist,
	trihandle*& outtet2trilist,
	trihandle*& outtet2seglist,
	bool verbose
)
{
	int numofpoint = input_mesh->numofpoints;
	double* pointlist = input_mesh->pointlist;
	int numofedge = input_mesh->numofedges;
	int* edgelist = input_mesh->edgelist;
	int numoftriface = input_mesh->numoftrifaces;
	int* trifacelist = input_mesh->trifacelist;
	int numoftet = input_mesh->numoftets;
	int* tetlist = input_mesh->tetlist;

	int* intertri2tetlist = input_mesh->interneigh ? input_mesh->intertri2tetlist : NULL;
	int* intertri2trilist = input_mesh->interneigh ? input_mesh->intertri2trilist : NULL;
	int* intertet2tetlist = input_mesh->interneigh ? input_mesh->intertet2tetlist : NULL;

	// Initialization
	inittables();

	std::vector<int> outtetvector(4 * numoftet);
	memcpy(outtetvector.data(), tetlist, 4 * numoftet * sizeof(int));
	tethandle * point2tet = new tethandle[numofpoint];
	verttype * pointtype = new verttype[numofpoint];
	std::vector<tethandle> outneighborvector(4 * numoftet, tethandle(-1, 11));

	// Initialize point2tet list
	for (int i = 0; i < numofpoint; i++)
	{
		point2tet[i] = tethandle(-1, 11);
	}

	// Initialize point type
	for (int i = 0; i < numofpoint; i++)
	{
		pointtype[i] = VOLVERTEX; // initial type
	}

	clock_t tv[2];

	// Create the tetrahedra and connect those that share a common face
	if (verbose)
	{
		printf("1. Create neighbor information.\n");
		tv[0] = clock();
	}
	if (intertet2tetlist == NULL)
	{
		tethandle * tmplist = new tethandle[4 * numoftet];
		for (int i = 0; i < numoftet; i++)
		{
			// Get the four vertices
			double* p[4]; // point pointers
			for (int j = 0; j < 4; j++)
			{
				// Get and check vertex index
				int idx = outtetvector[i * 4 + j]; // vertex index
												   // Get vertex pointer
				p[j] = id2pointlist(idx, pointlist);
			}

			// Check the orientation
			double ori = orient3d(p[0], p[1], p[2], p[3]);
			if (false) // assume all Tets are oriented correctly
					   //if (ori > 0)
			{
				// Swap the first two vertices in tetrahedron list
				// if this tet is nearly degenerate, swap may be wrong
				// but still swap to try to orient the tet correctly
				int tmp;
				tmp = outtetvector[i * 4 + 0];
				outtetvector[i * 4 + 0] = outtetvector[i * 4 + 1];
				outtetvector[i * 4 + 1] = tmp;

				//if (isDegenerateTet(p[0], p[1], p[2], p[3]))
				//{
				//	printf("Nearly degenerate Tet #%d - Ori %e - %d, %d, %d, %d\n",
				//		i, ori,
				//		outtetvector[4 * i + 0],
				//		outtetvector[4 * i + 1],
				//		outtetvector[4 * i + 2],
				//		outtetvector[4 * i + 3]);
				//}
				//printf("Swap Tet #%d - Ori %e - %d, %d, %d, %d\n",
				//	i, ori,
				//	outtetvector[4 * i + 0],
				//	outtetvector[4 * i + 1],
				//	outtetvector[4 * i + 2],
				//	outtetvector[4 * i + 3]);
			}
			else if (ori == 0.0)
			{
				if (verbose)
					printf("Warning: Tet #%d is degenerate.\n", i);
			}

			// Make TetHandle for this tetrahedron
			tethandle tetloop(i, 11); // tetloop.ver = 11;

									  // Try to connect to neighbors
			for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++)
			{
				int p[4], q[4];
				tethandle tmphandle(-1, 11), checktet, prevchktet;
				// Look for other tets having this vertex
				p[3] = oppo(tetloop, outtetvector.data());
				tmphandle = point2tet[p[3]];
				// Link the current tet to the existing one
				tmplist[4 * tetloop.id + tetloop.ver] = tmphandle;
				// Update current tet to point
				point2tet[p[3]] = tetloop;
				checktet = tmphandle;
				if (checktet.id != -1) // not empty handle
				{
					p[0] = org(tetloop, outtetvector.data()); // a
					p[1] = dest(tetloop, outtetvector.data()); // b
					p[2] = apex(tetloop, outtetvector.data()); // c
					prevchktet = tetloop;
					do {
						q[0] = org(checktet, outtetvector.data());  // a'
						q[1] = dest(checktet, outtetvector.data()); // b'
						q[2] = apex(checktet, outtetvector.data()); // c'
																	// Check the three faces at 'd' in 'checktet'
						int bondflag = 0;
						for (int j = 0; j < 3; j++)
						{
							// Go to the face [b',a',d'] or [c',b',d] or [a',c',d].
							tethandle face1, face2;
							esym(checktet, face2);
							if (outneighborvector[4 * face2.id + (face2.ver & 3)].id == -1) // empty neighbor
							{
								int k = ((j + 1) % 3);
								if (q[k] == p[0])     // b', c', a' = a
								{
									if (q[j] == p[1]) // a', b', c' = b
									{
										// [#,#,d] is matched to [b,a,d].
										esym(tetloop, face1);
										bond(face1, face2, outneighborvector.data());
										bondflag++;
									}
								}
								if (q[k] == p[1])	  // b',c',a' = b
								{
									if (q[j] == p[2]) // a',b',c' = c
									{
										// [#,#,d] is matched to [c,b,d].
										enext(tetloop, face1);
										esymself(face1);
										bond(face1, face2, outneighborvector.data());
										bondflag++;
									}
								}
								if (q[k] == p[2])	  // b',c',a' = c
								{
									if (q[j] == p[0]) // a',b',c' = a
									{
										// [#,#,d] is matched to [a,c,d].
										eprev(tetloop, face1);
										esymself(face1);
										bond(face1, face2, outneighborvector.data());
										bondflag++;
									}
								}
							}
							else // already has neighbor
							{
								bondflag++;
							}
							enextself(checktet);
						} // j
						  // Go to the next tet in the link.
						tethandle tptr = tmplist[4 * checktet.id + checktet.ver]; // 0 <= checktet.ver <= 3
						if (bondflag == 3)
						{
							// All three faces at d in 'checktet' have been connected.
							// It can be removed from the link
							tmplist[4 * prevchktet.id + prevchktet.ver] = tptr; // 0 <= prevchket.ver <= 3
						}
						else
						{
							// Backup the previous tet in the link
							prevchktet = checktet;
						}
						checktet = tptr;
					} while (checktet.id != -1);
				} // if (checktet.id != -1)
			} // for(tetloop.ver = 0; ...
		} // i
		delete[] tmplist;
	}
	else
	{
		int p, neighidx, neighver;
		tethandle tetloop, checktet;
		for (int i = 0; i < numoftet; i++)
		{
			tetloop.id = i;
			for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++)
			{
				p = oppo(tetloop, outtetvector.data()); // d
				point2tet[p] = tetloop;

				if (outneighborvector[4 * tetloop.id + tetloop.ver].id != -1)
					continue;

				neighidx = intertet2tetlist[8 * tetloop.id + 2 * tetloop.ver];
				if (neighidx == -1)
					continue;
				neighver = intertet2tetlist[8 * tetloop.id + 2 * tetloop.ver + 1];

				checktet.id = neighidx;
				checktet.ver = neighver;

				bond(tetloop, checktet, outneighborvector.data());
			}
		}
	}
	//else
	//{
	//	int p[4], q[4], neighidx;
	//	bool found = false;
	//	tethandle tetloop, checktet;
	//	for (int i = 0; i < numoftet; i++)
	//	{
	//		tetloop.id = i;
	//		for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++)
	//		{
	//			p[3] = oppo(tetloop, outtetvector.data()); // d
	//			point2tet[p[3]] = tetloop;

	//			if (outneighborvector[4 * tetloop.id + tetloop.ver].id != -1)
	//				continue;

	//			neighidx = neighborlist[4 * tetloop.id + tetloop.ver];
	//			if (neighidx == -1)
	//				continue;

	//			p[0] = org(tetloop, outtetvector.data()); // a
	//			p[1] = dest(tetloop, outtetvector.data()); // b
	//			p[2] = apex(tetloop, outtetvector.data()); // c

	//			// let checktet oppo become 'c'
	//			checktet.id = neighidx;
	//			for (checktet.ver = 0; checktet.ver < 4; checktet.ver++)
	//			{
	//				q[3] = oppo(checktet, outtetvector.data());
	//				if (q[3] == p[2])
	//				{
	//					found = true;
	//					break;
	//				}
	//			}
	//			assert(found);
	//			found = false;

	//			// let checktet share the edge org->dest with tetloop
	//			for (int j = 0; j < 3; j++)
	//			{
	//				q[0] = org(checktet, outtetvector.data());
	//				q[1] = dest(checktet, outtetvector.data());
	//				if (q[0] == p[0] && q[1] == p[1]) // found the edge
	//				{
	//					esymself(checktet);
	//					bond(tetloop, checktet, outneighborvector.data());
	//					found = true;
	//					break;
	//				}
	//				enextself(checktet);
	//			}
	//			assert(found);
	//			found = false;
	//		}
	//	}
	//}

	// debug
	//for (int i = 109000; i < 110000; i++)
	//{
	//	printf("Tet#%d - ", i);
	//	for (int j = 0; j < 4; j++)
	//	{
	//		tethandle tmp = outneighborvector[4 * i + j];
	//		printf("%d(%d) ", tmp.id, tmp.ver);
	//	}
	//	printf("\n");
	//}

	if (verbose)
	{
		tv[1] = clock();
		printf("time: %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Create hull tets
	if(verbose)
		printf("2. Create hull tets\n");
	int hullsize, outnumoftet = numoftet;
	int count = 0;
	while (count < outnumoftet)
	{
		tethandle tetloop(count, 11);
		tethandle tptr = tetloop;
		for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++)
		{
			if (outneighborvector[4 * tetloop.id + tetloop.ver].id == -1) // empty neighbor
			{
				// Create a hull tet.
				outneighborvector.push_back(tethandle(-1, 11));
				outneighborvector.push_back(tethandle(-1, 11));
				outneighborvector.push_back(tethandle(-1, 11));
				outneighborvector.push_back(tethandle(-1, 11));
				tethandle hulltet(outnumoftet, 11);
				int p[3];
				p[0] = org(tetloop, outtetvector.data());
				p[1] = dest(tetloop, outtetvector.data());
				p[2] = apex(tetloop, outtetvector.data());
				outtetvector.push_back(p[1]);
				outtetvector.push_back(p[0]);
				outtetvector.push_back(p[2]);
				outtetvector.push_back(-1);
				outnumoftet++;
				bond(tetloop, hulltet, outneighborvector.data());
				// Try connecting this to others that share common hull edges.
				for (int j = 0; j < 3; j++)
				{
					tethandle face1, face2;
					fsym(hulltet, face2, outneighborvector.data());
					while (1)
					{
						if (face2.id == -1)
							break;
						esymself(face2);
						if (apex(face2, outtetvector.data()) == -1)
							break;
						fsymself(face2, outneighborvector.data());
					}
					if (face2.id != -1)
					{
						// Found an adjacent hull tet.
						esym(hulltet, face1);
						bond(face1, face2, outneighborvector.data());
					}
					enextself(hulltet);
				}
			}
			// Update the point-to-tet map.
			int idx = outtetvector[4 * tetloop.id + tetloop.ver];
			if (idx != -1) // not dummpy point
				point2tet[idx] = tptr;
		}
		count++;
	}
	hullsize = outnumoftet - numoftet;
	if (verbose)
	{
		printf("Hull size = %d\n", hullsize);
		tv[0] = clock();
		printf("time: %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Subfaces will be inserted into the mesh
	if(verbose)
		printf("3. Insert subfaces\n");
	std::vector<int> markervector(outnumoftet, 0);
	std::vector<trihandle> outtet2trivector(4 * outnumoftet, trihandle(-1, 0));
	std::vector<tethandle> outtri2tetvector;
	std::vector<int> outtrifacevector;
	trihandle * point2sh = new trihandle[numofpoint];
	int outnumofface;
	count = 0; // number of triface in output
	if (intertri2tetlist == NULL)
	{
		for (int i = 0; i < numoftriface; i++)
		{
			// Variables
			int j;
			int p[3];
			tethandle checktet, tetloop;
			trihandle neighsh, subloop;
			// Get endpoints
			for (j = 0; j < 3; j++)
			{
				p[j] = trifacelist[3 * i + j];
			}
			// Search the subface.
			int bondflag = 0;
			// Make sure all vertices are in the mesh. Avoid crash.
			for (j = 0; j < 3; j++)
			{
				checktet = point2tet[p[j]];
				if (checktet.id == -1)
				{
					//printf("break ");
					break;
				}
			}
			if ((j == 3) && getedge(p[0], p[1], &checktet, point2tet, pointlist, outtetvector.data(), outneighborvector.data(), markervector.data())) {
				tetloop = checktet;
				int q = apex(checktet, outtetvector.data());
				while (1) {
					if (apex(tetloop, outtetvector.data()) == p[2]) {
						// Found the face.
						// Check if there exist a subface already?
						tspivot(tetloop, neighsh, outtet2trivector.data());
						if (neighsh.id != -1) {
							// Found a duplicated subface. 
							// This happens when the mesh was generated by other mesher.
							bondflag = 0;
						}
						else {
							bondflag = 1;
						}
						break;
					}
					fnextself(tetloop, outneighborvector.data());
					if (apex(tetloop, outtetvector.data()) == q)
					{
						break;
					}
				}
			}
			else
			{
				if (verbose)
					printf("Warning:  Edge [%d,%d] is missing.\n", p[0], p[1]);
			}

			if (bondflag) {
				// Create a new subface.
				subloop.id = count;
				subloop.shver = 0;
				outtrifacevector.push_back(p[0]);
				outtrifacevector.push_back(p[1]);
				outtrifacevector.push_back(p[2]);
				outtri2tetvector.push_back(tethandle(-1, 11));
				outtri2tetvector.push_back(tethandle(-1, 11));
				// Create the point-to-subface map.
				for (j = 0; j < 3; j++) {
					pointtype[p[j]] = FACETVERTEX; // initial type.
					point2sh[p[j]] = subloop;
				}
				// Insert the subface into the mesh.
				tsbond(tetloop, subloop, outtet2trivector.data(), outtri2tetvector.data());
				fsymself(tetloop, outneighborvector.data());
				sesymself(subloop);
				tsbond(tetloop, subloop, outtet2trivector.data(), outtri2tetvector.data());
				count++;
			}
			else {
				if (neighsh.id == -1) {
					if (verbose)
						printf("Warning:  Subface #%d [%d,%d,%d] is missing.\n",
							i, p[0], p[1], p[2]);
				}
				else {
					if (verbose)
						printf("Warning: Ignore a dunplicated subface #%d [%d,%d,%d].\n",
							i, p[0], p[1], p[2]);
				}
			} // if (bondflag)
		} // i
	}
	else
	{
		for (int i = 0; i < numoftriface; i++)
		{
			// Variables
			int j;
			int p[3], q[2];
			int neigh, ver;
			tethandle checktet, tetloop;
			trihandle neighsh, subloop;
			// Get endpoints
			for (j = 0; j < 3; j++)
			{
				p[j] = trifacelist[3 * i + j];
			}
			// Search the subface.
			int bondflag = 0;
			// Make sure all vertices are in the mesh. Avoid crash.
			for (j = 0; j < 3; j++)
			{
				checktet = point2tet[p[j]];
				if (checktet.id == -1)
				{
					break;
				}
			}
			if (j != 3)
			{
				if (verbose)
					printf("Warning:  Edge [%d,%d] is missing.\n", p[0], p[1]);
				continue;
			}
			neigh = intertri2tetlist[2 * i + 0];
			assert(neigh1 != -1);
			ver = intertri2tetlist[2 * i + 1];
			tetloop.id = neigh;
			tetloop.ver = ver;
			bool found = false;
			for (j = 0; j < 3; j++)
			{
				if (apex(tetloop, outtetvector.data()) == p[2]) {
					q[0] = org(tetloop, outtetvector.data());
					q[1] = dest(tetloop, outtetvector.data());
					if ((q[0] == p[0] && q[1] == p[1]) || (q[0] == p[1] && q[1] == p[0]))
					{
						found = true;
						// Found the face.
						// Check if there exist a subface already?
						tspivot(tetloop, neighsh, outtet2trivector.data());
						if (neighsh.id != -1) {
							// Found a duplicated subface. 
							// This happens when the mesh was generated by other mesher.
							bondflag = 0;
						}
						else {
							if (q[0] == p[1])
								fsymself(tetloop, outneighborvector.data());
							bondflag = 1;
						}
						break;
					}
				}
				enextself(tetloop);
			}
			if (!found)
			{
				// The information in intertri2tetlist may be wrong.
				// Search the tet again
				if (getedge(p[0], p[1], &checktet, point2tet, pointlist, outtetvector.data(), outneighborvector.data(), markervector.data())) {
					tetloop = checktet;
					int q = apex(checktet, outtetvector.data());
					while (1) {
						if (apex(tetloop, outtetvector.data()) == p[2]) {
							// Found the face.
							// Check if there exist a subface already?
							tspivot(tetloop, neighsh, outtet2trivector.data());
							if (neighsh.id != -1) {
								// Found a duplicated subface. 
								// This happens when the mesh was generated by other mesher.
								bondflag = 0;
							}
							else {
								bondflag = 1;
							}
							break;
						}
						fnextself(tetloop, outneighborvector.data());
						if (apex(tetloop, outtetvector.data()) == q)
						{
							break;
						}
					}
				}
				else
				{
					if (verbose)
						printf("Warning:  Edge [%d,%d] is missing.\n", p[0], p[1]);
				}
			}

			// bind suface and tetrahedra
			if (bondflag) {
				// Create a new subface.
				subloop.id = count;
				subloop.shver = 0;
				outtrifacevector.push_back(p[0]);
				outtrifacevector.push_back(p[1]);
				outtrifacevector.push_back(p[2]);
				outtri2tetvector.push_back(tethandle(-1, 11));
				outtri2tetvector.push_back(tethandle(-1, 11));
				// Create the point-to-subface map.
				for (j = 0; j < 3; j++) {
					pointtype[p[j]] = FACETVERTEX; // initial type.
					point2sh[p[j]] = subloop;
				}
				// Insert the subface into the mesh.
				tsbond(tetloop, subloop, outtet2trivector.data(), outtri2tetvector.data());
				fsymself(tetloop, outneighborvector.data());
				sesymself(subloop);
				tsbond(tetloop, subloop, outtet2trivector.data(), outtri2tetvector.data());
				count++;
			}
			else {
				if (neighsh.id == -1) {
					if (verbose)
						printf("Warning:  Subface #%d [%d,%d,%d] is missing.\n",
							i, p[0], p[1], p[2]);
				}
				else {
					if (verbose)
						printf("Warning: Ignore a dunplicated subface #%d [%d,%d,%d].\n",
							i, p[0], p[1], p[2]);
				}
			} // if (bondflag)
		}
	}

	outnumofface = count;
	if (verbose)
	{
		printf("Inserted subfaces size = %d\n", outnumofface);
		tv[1] = clock();
		printf("time: %f\n", (REAL)(tv[1] - tv[0]));
	}

	//for (int i = 0; i < outnumofface; i++)
	//{
	//	printf("Subface #%d - %d, %d\n",
	//		i, outtri2tetvector[2 * i].id, outtri2tetvector[2 * i + 1].id);
	//}

	// Indentify subfaces from the mesh.
	// Create subfaces for hull faces (if they're not subface yet).
	// Input mesh is not convex by default
	if(verbose)
		printf("4. Create subfaces for hull faces\n");
	count = 0;
	while (count < outnumoftet)
	{
		tethandle tetloop(count, 11), checktet;
		trihandle neighsh;
		int bondflag;
		int p[3];
		trihandle subloop;
		for (tetloop.ver = 0; tetloop.ver < 4; tetloop.ver++) {
			tspivot(tetloop, neighsh, outtet2trivector.data());
			if (neighsh.id == -1) {
				bondflag = 0;
				fsym(tetloop, checktet, outneighborvector.data());
				if (!ishulltet(tetloop, outtetvector.data()) && // avoid create a subface between hull tets, need to check if this is correct
					ishulltet(checktet, outtetvector.data())) {
					// A hull face.
					if (true) { // not convex by default
						bondflag = 1;  // Insert a hull subface.
					}
				}
				if (bondflag) {
					//printf("Tet #%d - ver %d\n", tetloop.id, tetloop.ver);
					// Create a new subface.
					subloop.id = outnumofface;
					subloop.shver = 0;
					p[0] = org(tetloop, outtetvector.data());
					p[1] = dest(tetloop, outtetvector.data());
					p[2] = apex(tetloop, outtetvector.data());
					outtrifacevector.push_back(p[0]);
					outtrifacevector.push_back(p[1]);
					outtrifacevector.push_back(p[2]);
					outtri2tetvector.push_back(tethandle(-1, 11));
					outtri2tetvector.push_back(tethandle(-1, 11));
					// Create the point-to-subface map.
					for (int j = 0; j < 3; j++) {
						pointtype[p[j]] = FACETVERTEX; // initial type.
						point2sh[p[j]] = subloop;
					}
					// Insert the subface into the mesh.
					tsbond(tetloop, subloop, outtet2trivector.data(), outtri2tetvector.data());
					sesymself(subloop);
					tsbond(checktet, subloop, outtet2trivector.data(), outtri2tetvector.data());
					outnumofface++;
				} // if (bondflag)
			} // if (neighsh.id == -1)
		}
		count++;
	}
	if (verbose)
	{
		printf("Output subfaces size = %d\n", outnumofface);
		tv[0] = clock();
		printf("time: %f\n", (REAL)(tv[0] - tv[1]));
	}

	// Connect subfaces together.
	if(verbose)
		printf("5. Connect subfaces together\n");
	count = 0;
	std::vector<trihandle> outtri2trivector(3 * outnumofface, trihandle(-1, 0));
	if (intertri2trilist == NULL)
	{
		int q;
		while (count < outnumofface)
		{
			trihandle subloop(count, 0), neighsh, nextsh;
			tethandle tetloop;
			for (int i = 0; i < 3; i++) {
				spivot(subloop, neighsh, outtri2trivector.data());
				if (neighsh.id == -1) {
					// Form a subface ring by linking all subfaces at this edge.
					// Traversing all faces of the tets at this edge.
					stpivot(subloop, tetloop, outtri2tetvector.data());
					q = apex(tetloop, outtetvector.data());
					neighsh = subloop;
					while (1) {
						fnextself(tetloop, outneighborvector.data());
						tspivot(tetloop, nextsh, outtet2trivector.data());
						if (nextsh.id != -1) {
							// Link neighsh <= nextsh.
							sbond1(neighsh, nextsh, outtri2trivector.data());
							neighsh = nextsh;
						}
						if (apex(tetloop, outtetvector.data()) == q) {
							assert(nextsh.id == subloop.id); // It's a ring.
							break;
						}
					} // while (1)
				} // if (neighsh.id == -1)
				senextself(subloop);
			}
			count++;
		}
	}
	/*else
	{
		int p[2];
		bool found = false;
		while (count < outnumofface)
		{
			trihandle subloop(count, 0), neighsh;
			for (int i = 0; i < 3; i++) {
				neighsh.id = intertri2trilist[6 * count + 2 * i];
				if (neighsh.id != -1)
				{
					neighsh.shver = intertri2trilist[6 * count + 2 * i + 1];
					p[0] = sorg(subloop, outtrifacevector.data());
					p[1] = sdest(subloop, outtrifacevector.data());
					p[2] = sapex(subloop, outtrifacevector.data());
					for (int j = 0; j < 3; j++)
					{
						p[5] = sapex(neighsh, outtrifacevector.data());
						if (p[2] == p[5])
						{
							p[3] = sorg(neighsh, outtrifacevector.data());
							p[4] = sdest(neighsh, outtrifacevector.data());
							if ((p[0] == p[3] && p[1] == p[4]) || (p[0] == p[4] && p[1] == p[3]))
							{
								sbond1(subloop, neighsh, outtri2trivector.data());
								found = true;
								break;
							}
						}
						else
							senextself(neighsh);
					}
					assert(found);
					found = false;
				}
				senextself(subloop);
			}
			count++;
		}
	}*/

	if (verbose)
	{
		tv[1] = clock();
		printf("time: %f\n", (REAL)(tv[1] - tv[0]));
	}

	//for (int i = 0; i < outnumofface; i++)
	//{
	//	printf("Subface #%d - %d, %d, %d\n", i,
	//		outtri2trivector[3 * i + 0],
	//		outtri2trivector[3 * i + 1],
	//		outtri2trivector[3 * i + 2]);
	//}

	// Segments will be introduced.
	if(verbose)
		printf("6. Insert segments\n");
	int outnumofseg;
	std::vector<trihandle> outtet2segvector(6 * outnumoftet, trihandle(-1, 0));
	std::vector<trihandle> outtri2segvector(3 * outnumofface, trihandle(-1, 0));
	std::vector<trihandle> outseg2trivector;
	std::vector<tethandle> outseg2tetvector;
	std::vector<int> outsegvector;
	count = 0;
	for (int i = 0; i < numofedge; i++)
	{
		int j;
		int p[2], q;
		tethandle checktet, tetloop;
		// Insert a segment.
		for (j = 0; j < 2; j++) {
			p[j] = edgelist[i * 2 + j];
		}
		// Make sure all vertices are in the mesh. Avoid crash.
		for (j = 0; j < 2; j++) {
			checktet = point2tet[p[j]];
			if (checktet.id == -1) break;
		}
		// Search the segment.
		if ((j == 2) && getedge(p[0], p[1], &checktet, point2tet, pointlist, outtetvector.data(), outneighborvector.data(), markervector.data())) {
			// Create a new subface.
			trihandle segloop(count, 0);
			outsegvector.push_back(p[0]);
			outsegvector.push_back(p[1]);
			outsegvector.push_back(-1);
			outseg2trivector.push_back(trihandle(-1, 0));
			outseg2trivector.push_back(trihandle(-1, 0));
			outseg2trivector.push_back(trihandle(-1, 0));
			outseg2tetvector.push_back(tethandle(-1, 11));
			// Create the point-to-segment map.
			for (j = 0; j < 2; j++) {
				pointtype[p[j]] = RIDGEVERTEX; // initial type.
				point2sh[p[j]] = segloop;
			}
			// Insert the segment into the mesh.
			tetloop = checktet;
			q = apex(checktet, outtetvector.data());
			trihandle subloop(-1, 0);
			while (1) {
				tssbond1(tetloop, segloop, outtet2segvector.data());
				tspivot(tetloop, subloop, outtet2trivector.data());
				if (subloop.id != -1) {
					ssbond1(subloop, segloop, outtri2segvector.data());
					sbond1(segloop, subloop, outseg2trivector.data());
				}
				fnextself(tetloop, outneighborvector.data());
				if (apex(tetloop, outtetvector.data()) == q) break;
			} // while (1)
			// Remember an adjacent tet for this segment.
			sstbond1(segloop, tetloop, outseg2tetvector.data());
			count++;
		}
		else {
			if(verbose)
				printf("Warning:  Segment #%d [%d,%d] is missing.\n",
					i, p[0], p[1]);
		}
	}
	outnumofseg = count;
	if (verbose)
	{
		printf("Inserted segment size = %d\n", outnumofseg);
		tv[0] = clock();
		printf("time: %f\n", (REAL)(tv[0] - tv[1]));
	}

	//for (int i = 0; i < outnumofseg; i++)
	//{
	//	printf("Seg #%d - %d, %d, %d - %d, %d, %d - %d, %d\n",
	//		i, outsegvector[3 * i], outsegvector[3 * i + 1], outsegvector[3 * i + 2],
	//		outseg2trivector[3 * i].id, outseg2trivector[3 * i + 1].id, outseg2trivector[3 * i + 2].id,
	//		outseg2tetvector[2 * i].id, outseg2tetvector[2 * i + 1].id);
	//}
	//for (int i = 0; i < outnumoftet; i++)
	//{
	//	printf("Tet #%d - %d, %d, %d, %d, %d, %d\n",
	//		i, outtet2segvector[6 * i].id, outtet2segvector[6 * i + 1].id,
	//		outtet2segvector[6 * i + 2].id, outtet2segvector[6 * i + 3].id,
	//		outtet2segvector[6 * i + 4].id, outtet2segvector[6 * i + 5].id);
	//}
	//for (int i = 0; i < outnumofface; i++)
	//{
	//	printf("Subface #%d - %d, %d, %d\n", i,
	//		outtri2segvector[3 * i].id, outtri2segvector[3 * i + 1].id, outtri2segvector[3 * i + 2].id);
	//}

	// Identify segments from the mesh. 
	// Create segments for non-manifold edges (which are shared by more 
	//   than two subfaces).
	if(verbose)
		printf("7. Create segments for non-manifold edges\n");
	count = 0;
	REAL ang, angtol = 179.9 / 180.0 * PI;
	while (count < outnumofface)
	{
		trihandle subloop(count, 0), segloop;
		trihandle nextsh, neighsh;
		tethandle tetloop;
		int bondflag, idx;
		int p[4], q;
		for (int i = 0; i < 3; i++) {
			sspivot(subloop, segloop, outtri2segvector.data());
			if (segloop.id == -1) {
				// Check if this edge is a segment.
				bondflag = 0;
				// Counter the number of subfaces at this edge.
				idx = 0;
				nextsh = subloop;
				while (1) {
					idx++;
					spivotself(nextsh, outtri2trivector.data());
					if (nextsh.id == subloop.id) break;
				}
				if (idx != 2) {
					// It's a non-manifold edge. Insert a segment.
					p[0] = sorg(subloop, outtrifacevector.data());
					p[1] = sdest(subloop, outtrifacevector.data());
					bondflag = 1;
				}
				else // need to check if this part is necessary
				{
					//spivot(subloop, neighsh, outtri2trivector.data());
					//if (true) // not convex by default
					//{
					//	// Check the dihedral angle formed by the two subfaces.
					//	p[0] = sorg(subloop, outtrifacevector.data());
					//	p[1] = sdest(subloop, outtrifacevector.data());
					//	p[2] = sapex(subloop, outtrifacevector.data());
					//	p[3] = sapex(neighsh, outtrifacevector.data());
					//	REAL* cp[4];
					//	for (int j = 0; j < 4; j++)
					//	{
					//		cp[j] = id2pointlist(p[j], pointlist);
					//	}
					//	ang = facedihedral(cp[0], cp[1], cp[2], cp[3]);
					//	if (ang > PI) ang = 2 * PI - ang;
					//	if (ang < angtol) {
					//		bondflag = 1;
					//		//printf("Subface #%d, #%d - ang = %g, angtol = %g\n", subloop.id, neighsh.id, ang, angtol);
					//	}
					//}
				}

				if (bondflag) {
					// Create a new segment.
					trihandle segloop(outnumofseg, 0);
					outsegvector.push_back(p[0]);
					outsegvector.push_back(p[1]);
					outsegvector.push_back(-1);
					outseg2trivector.push_back(trihandle(-1, 0));
					outseg2trivector.push_back(trihandle(-1, 0));
					outseg2trivector.push_back(trihandle(-1, 0));
					outseg2tetvector.push_back(tethandle(-1, 11));
					// Create the point-to-segment map.
					for (int j = 0; j < 2; j++) {
						pointtype[p[j]] = RIDGEVERTEX; // initial type.
						point2sh[p[j]] = segloop;
					}
					// Insert the subface into the mesh.
					stpivot(subloop, tetloop, outtri2tetvector.data());
					q = apex(tetloop, outtetvector.data());
					while (1) {
						tssbond1(tetloop, segloop, outtet2segvector.data());
						tspivot(tetloop, neighsh, outtet2trivector.data());
						if (neighsh.id != -1) {
							ssbond1(neighsh, segloop, outtri2segvector.data());
						}
						fnextself(tetloop, outneighborvector.data());
						if (apex(tetloop, outtetvector.data()) == q) break;
					} // while (1)
					  // Remember an adjacent tet for this segment.
					sstbond1(segloop, tetloop, outseg2tetvector.data());
					sbond1(segloop, subloop, outseg2trivector.data());
					outnumofseg++;
				} // if (bondflag)
			} // if (neighsh.id == -1)
			senextself(subloop);
		} // i
		count++;
	}
	if (verbose)
	{
		printf("Output segment size = %d\n", outnumofseg);
		tv[1] = clock();
		printf("time: %f\n", (REAL)(tv[1] - tv[0]));
	}

	// Mark Steiner points on segments and facets.
	//   - all vertices which remaining type FEACTVERTEX become
	//     Steiner points in facets (= FREEFACETVERTEX).
	//   - vertices on segment need to be checked.
	if(verbose)
		printf("8. Mark Steiner points on segments and facets\n");
	trihandle* segperverlist;
	int* idx2seglist;
	makepoint2submap(outsegvector.data(), idx2seglist, segperverlist, outnumofseg, numofpoint);
	count = 0;
	while (count < numofpoint)
	{
		verttype vt = pointtype[count];
		if (vt == VOLVERTEX)
		{
			pointtype[count] = FREEVOLVERTEX;
		}
		else if (vt == FACETVERTEX)
		{
			pointtype[count] = FREEFACETVERTEX;
		}
		else if (vt == RIDGEVERTEX)
		{
			int idx = count;
			if ((idx2seglist[idx + 1] - idx2seglist[idx]) == 2)
			{
				int i = idx2seglist[idx];
				trihandle parentseg = segperverlist[i];
				trihandle nextseg = segperverlist[i + 1];
				sesymself(nextseg);
				int p[2];
				p[0] = sorg(nextseg, outsegvector.data());
				p[1] = sdest(parentseg, outsegvector.data());
				double *pc[3];
				pc[0] = id2pointlist(p[0], pointlist);
				pc[1] = id2pointlist(p[1], pointlist);
				pc[2] = id2pointlist(count, pointlist);
				// Check if three points p[0], ptloop, p[2] are (nearly) collinear.
				REAL len, l1, l2;
				len = pointdistance(pc[0], pc[1]);
				l1 = pointdistance(pc[0], pc[2]);
				l2 = pointdistance(pc[2], pc[1]);
				if (((l1 + l2 - len) / len) < EPSILON) {
					// They are (nearly) collinear.
					pointtype[count] = FREESEGVERTEX;
					// Connect nextseg and parentseg together at ptloop.
					senextself(nextseg);
					senext2self(parentseg);
					sbond(nextseg, parentseg, outseg2trivector.data());
				}
			}
		}
		count++;
	}
	if (verbose)
	{
		tv[0] = clock();
		printf("time: %f\n", (REAL)(tv[0] - tv[1]));
	}

	delete[] idx2seglist;
	delete[] segperverlist;

	//for (int i = 0; i < outnumofseg; i++)
	//{
	//	printf("Seg #%d - %d, %d, %d\n", i,
	//		outseg2trivector[3 * i].id,
	//		outseg2trivector[3 * i + 1].id,
	//		outseg2trivector[3 * i + 2].id);
	//}

	// Output all the needed information
	outpoint2tetlist = point2tet;
	outpoint2trilist = point2sh;
	outpointtypelist = pointtype;
	outnumofedge = outnumofseg;
	outseglist = new int[3 * outnumofedge];
	std::copy(outsegvector.begin(), outsegvector.end(), outseglist);
	outseg2trilist = new trihandle[3 * outnumofedge];
	std::copy(outseg2trivector.begin(), outseg2trivector.end(), outseg2trilist);
	outseg2tetlist = new tethandle[outnumofedge];
	std::copy(outseg2tetvector.begin(), outseg2tetvector.end(), outseg2tetlist);
	outnumoftriface = outnumofface;
	outtrifacelist = new int[3 * outnumoftriface];
	std::copy(outtrifacevector.begin(), outtrifacevector.end(), outtrifacelist);
	outtri2tetlist = new tethandle[2 * outnumoftriface];
	std::copy(outtri2tetvector.begin(), outtri2tetvector.end(), outtri2tetlist);
	outtri2trilist = new trihandle[3 * outnumoftriface];
	std::copy(outtri2trivector.begin(), outtri2trivector.end(), outtri2trilist);
	outtri2seglist = new trihandle[3 * outnumoftriface];
	std::copy(outtri2segvector.begin(), outtri2segvector.end(), outtri2seglist);
	outnumoftetrahedron = outnumoftet;
	outtetlist = new int[4 * outnumoftetrahedron];
	std::copy(outtetvector.begin(), outtetvector.end(), outtetlist);
	outneighborlist = new tethandle[4 * outnumoftetrahedron];
	std::copy(outneighborvector.begin(), outneighborvector.end(), outneighborlist);
	outtet2trilist = new trihandle[4 * outnumoftetrahedron];
	std::copy(outtet2trivector.begin(), outtet2trivector.end(), outtet2trilist);
	outtet2seglist = new trihandle[6 * outnumoftetrahedron];
	std::copy(outtet2segvector.begin(), outtet2segvector.end(), outtet2seglist);
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// makesegment2parentmap()    Create a map from a segment to its parent.     //
//                                                                           //
// The map is saved in the array 'segment2parentlist' and					 //
// 'segmentendpointslist'. 													 //
// The length of 'segmentendpointslist'	is twice the number of segments.     //
// Each segment is assigned a unique index (starting from 0).                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void makesegment2parentmap(
	int numofsegment,
	int* segmentlist,
	trihandle* seg2trilist,
	int*& segment2parentlist,
	int*& segmentendpointslist,
	int& numofparent
)
{
	trihandle segloop, prevseg, nextseg;
	int eorg, edest;
	int segindex = 0, idx = 0;
	int i, count;

	segment2parentlist = new int[numofsegment];
	std::vector<int> segptvec;

	// A segment s may have been split into many subsegments. Operate the one
	//   which contains the origin of s. Then mark the rest of subsegments.
	segloop.id = count = 0;
	segloop.shver = 0;
	while (count < numofsegment) {
		senext2(segloop, prevseg);
		spivotself(prevseg, seg2trilist);
		if (prevseg.id == -1) {
			eorg = sorg(segloop, segmentlist);
			edest = sdest(segloop, segmentlist);
			segment2parentlist[segloop.id] = segindex;
			senext(segloop, nextseg);
			spivotself(nextseg, seg2trilist);
			while (nextseg.id != -1) {
				segment2parentlist[nextseg.id] = segindex;
				nextseg.shver = 0;
				if (sorg(nextseg, segmentlist) != edest) sesymself(nextseg);
				assert(sorg(nextseg) == edest);
				edest = sdest(nextseg, segmentlist);
				// Go the next connected subsegment at edest.
				senextself(nextseg);
				spivotself(nextseg, seg2trilist);
			}
			segptvec.push_back(eorg);
			segptvec.push_back(edest);
			segindex++;
		}
		segloop.id = ++count;
	}

	segmentendpointslist = new int[2 * segindex];
	std::copy(segptvec.begin(), segptvec.end(), segmentendpointslist);
	numofparent = segindex;

	// debug
	//for (i = 0; i < numofsegment; i++)
	//{
	//	printf("seg #%d - %d\n", i, segment2parentlist[i]);
	//}
	//for (i = 0; i < segindex; i++)
	//{
	//	printf("seg parent #%d - %d, %d\n", i, segmentendpointslist[2 * i], segmentendpointslist[2 * i + 1]);
	//}
}

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// makesubfacepointsmap()    Create a map from facet to its vertices.        //
//                                                                           //
// All facets will be indexed (starting from 0).  The map is saved in three  //
// arrays: 'subface2parentlist', 'id2subfacelist', and 'subfacepointslist'.  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void makesubfacepointsmap(
	int numofpoint,
	double* pointlist,
	verttype* pointtypelist,
	int numofsubface,
	int* subfacelist,
	trihandle* subface2seglist,
	trihandle* subface2subfacelist,
	int*& subface2parentlist,
	int*& id2subfacelist,
	int*& subfacepointslist,
	int& numofparent,
	int& numofendpoints
)
{
	trihandle subloop, neighsh, parysh, parysh1;
	int pa, *ppt, parypt;
	verttype vt;
	int facetindex, totalvertices;
	int i, j, k, count;

	std::vector<std::vector<int>> facetvertexlist;
	facetindex = totalvertices = 0;
	std::vector<int> pmarker(numofpoint, 0);
	std::vector<int> smarker(numofsubface, 0);

	subface2parentlist = new int[numofsubface];

	subloop.id = count = 0;
	std::vector<trihandle> shlist;
	std::vector<int> vertlist;
	while (count < numofsubface) {
		if (smarker[subloop.id] == 0) {
			// A new facet. Create its vertices list.
			vertlist.clear();
			ppt = subfacelist + 3 * subloop.id;
			for (k = 0; k < 3; k++) {
				vt = pointtypelist[ppt[k]];
				if ((vt != FREESEGVERTEX) && (vt != FREEFACETVERTEX)) {
					pmarker[ppt[k]] = 1;
					vertlist.push_back(ppt[k]);
				}
			}
			smarker[subloop.id] = 1;
			shlist.push_back(subloop);
			for (i = 0; i < shlist.size(); i++) {
				parysh = shlist[i];
				subface2parentlist[parysh.id] = facetindex;
				for (j = 0; j < 3; j++) {
					if (!isshsubseg(parysh, subface2seglist)) {
						spivot(parysh, neighsh, subface2subfacelist);
						assert(neighsh.sh != NULL);
						//if (subloop.id == 27)
						//{
						//	int tmp[4];
						//	REAL* ctmp[4];
						//	REAL ang;
						//	tmp[0] = sorg(parysh, subfacelist);
						//	tmp[1] = sdest(parysh, subfacelist);
						//	tmp[2] = sapex(parysh, subfacelist);
						//	tmp[3] = sapex(neighsh, subfacelist);
						//	for (k = 0; k < 4; k++)
						//	{
						//		ctmp[k] = id2pointlist(tmp[k], pointlist);
						//	}
						//	ang = facedihedral(ctmp[0], ctmp[1], ctmp[2], ctmp[3]);
						//	if (ang > PI)
						//		ang = 2 * PI - ang;
						//	printf("Subface #%d - Edge %d, %d - ang = %g\n",
						//		parysh.id, tmp[0], tmp[1], ang);
						//}
						if (smarker[neighsh.id] == 0) {
							pa = sapex(neighsh, subfacelist);
							if (pmarker[pa] == 0) {
								vt = pointtypelist[pa];
								if ((vt != FREESEGVERTEX) && (vt != FREEFACETVERTEX)) {
									pmarker[pa] = 1;
									vertlist.push_back(pa);
								}
							}
							smarker[neighsh.id] = 1;
							shlist.push_back(neighsh);
						}
					}
					senextself(parysh);
				}
				//if(subloop.id == 27)
				//	printf("Subface #%d - %d, %d, %d\n", parysh.id, subfacelist[3*parysh.id], subfacelist[3 * parysh.id+1], subfacelist[3 * parysh.id+2]);
			} // i
			totalvertices += vertlist.size();
			// Uninfect facet vertices.
			for (k = 0; k < vertlist.size(); k++) {
				parypt = vertlist[k];
				pmarker[parypt] = 0;
			}
			//if (vertlist.size() != 3)
			//	printf("triface #%d - %d - %d, %d, %d\n", count, vertlist.size(), subfacelist[3 * count], subfacelist[3 * count + 1], subfacelist[3 * count + 2]);
			shlist.clear();
			// Save this vertex list.
			facetvertexlist.push_back(vertlist);
			facetindex++;
		}
		subloop.id = ++count;
	}

	id2subfacelist = new int[facetindex + 1];
	subfacepointslist = new int[totalvertices];

	id2subfacelist[0] = 0;
	for (i = 0, k = 0; i < facetindex; i++) {
		vertlist = facetvertexlist[i];
		id2subfacelist[i + 1] = (id2subfacelist[i] + vertlist.size());
		for (j = 0; j < vertlist.size(); j++) {
			parypt = vertlist[j];
			subfacepointslist[k] = parypt;
			k++;
		}
		//if (vertlist.size() == 4)
		//	printf("Ret = %g\n",
		//		orient3d(
		//			id2pointlist(vertlist[0], pointlist), 
		//			id2pointlist(vertlist[1], pointlist),
		//			id2pointlist(vertlist[2], pointlist),
		//			id2pointlist(vertlist[3], pointlist))
		//	);
	}
	assert(k == totalvertices);

	numofparent = facetindex;
	numofendpoints = totalvertices;
}