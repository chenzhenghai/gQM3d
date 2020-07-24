#include <stdio.h>
#include <string>
#include <sstream>
#include <cassert>
#include <helper_timer.h>

#include "InputGenerator.h"
#include "InputPredicates.h"

// constant number
const int maxtrynum = 100000000;

// Random number generator, obtained from http://oldmill.uchicago.edu/~wilder/Code/random/
unsigned long z, w, jsr, jcong; // Seeds
double min = 0, max = 1000;

void randinit(unsigned long x_, double min_, double max_)
{
	z = x_; w = x_; jsr = x_; jcong = x_;
	min = min_;
	max = max_;
}

unsigned long znew()
{
	return (z = 36969 * (z & 0xfffful) + (z >> 16));
}

unsigned long wnew()
{
	return (w = 18000 * (w & 0xfffful) + (w >> 16));
}

unsigned long MWC()
{
	return ((znew() << 16) + wnew());
}

unsigned long SHR3()
{
	jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5));
}

unsigned long CONG()
{
	return (jcong = 69069 * jcong + 1234567);
}

unsigned long rand_int()         // [0,2^32-1]
{
	return ((MWC() ^ CONG()) + SHR3());
}

double random()     // [0,1]
{
	return ((double)rand_int() / (double(ULONG_MAX) + 1));
}

double next()
{
	const double val = min + (max - min) * random();
	return val;
}

// Generate a random point on different distribution
void generateSinglePoint(REAL* x, REAL* y, REAL* z, Distribution dist)
{
	assert((x != NULL && y != NULL && z != NULL));
	switch (dist)
	{
		case UniformDistribution:
		{
			*x = next();
			*y = next();
			*z = next();
		}
		break;
		case GaussianDistribution:
		{
			double x1, x2, x3, w;
			double tx, ty, tz;

			do {
				do {
					x1 = 2.0 * random() - 1.0;
					x2 = 2.0 * random() - 1.0;
					x3 = 2.0 * random() - 1.0;
					w = x1 * x1 + x2 * x2 + x3 * x3;
				} while (w >= 1.0);

				w = sqrt((-2.0 * log(w)) / w);
				tx = x1 * w;
				ty = x2 * w;
				tz = x3 * w;
			} while (tx < -3 || tx >= 3 || ty < -3 || ty >= 3 || tz < -3 || tz >= 3);

			*x = min + (max - min) * ((tx + 3.0) / 6.0);
			*y = min + (max - min) * ((ty + 3.0) / 6.0);
			*z = min + (max - min) * ((tz + 3.0) / 6.0);
		}
		break;
		case BallDistribution:
		{
			double d;

			do
			{
				*x = random() - 0.5;
				*y = random() - 0.5;
				*z = random() - 0.5;

				d = *x * *x + *y * *y + *z * *z;
			} while (d > 0.45*0.45);

			*x += 0.5;
			*y += 0.5;
			*z += 0.5;
			*x = *x * (max - min) + min;
			*y = *y * (max - min) + min;
			*z = *z * (max - min) + min;
		}
		break;
		case SphereDistribution:
		{
			//double a, b, c, d, l;

			//do 
			//{
			//	a = random() * 2.0 - 1.0;
			//	b = random() * 2.0 - 1.0;
			//	c = random() * 2.0 - 1.0;
			//	d = random() * 2.0 - 1.0;

			//	l = a * a + b * b + c * c + d * d;

			//} while (l >= 1.0);

			//*x = 2.0 * (b * d + a * c) / l * 0.45;
			//*y = 2.0 * (c * d - a * b) / l * 0.45;
			//*z = (a * a + d * d - b * b - c * c) / l * 0.45;
			//*x += 0.5;
			//*y += 0.5;
			//*z += 0.5;
			//*x = *x * (max - min) + min;
			//*y = *y * (max - min) + min;
			//*z = *z * (max - min) + min;

			double d;

			do
			{
				*x = random() - 0.5;
				*y = random() - 0.5;
				*z = random() - 0.5;

				d = *x * *x + *y * *y + *z * *z;
			} while (d > 0.45*0.45 || d < 0.2*0.2);

			*x += 0.5;
			*y += 0.5;
			*z += 0.5;
			*x = *x * (max - min) + min;
			*y = *y * (max - min) + min;
			*z = *z * (max - min) + min;
		}
		break;
		case GridDistribution:
		{
			double v[3];

			for (int i = 0; i < 3; ++i)
			{
				const double val = next();
				const double frac = val - floor(val);
				v[i] = (frac < 0.5) ? floor(val) : ceil(val);
				if (v[i] == min)
					v[i] = 1;
				else if (v[i] == max)
					v[i] = max - 1;
			}

			*x = v[0];
			*y = v[1];
			*z = v[2];
		}
		break;
		case ThinSphereDistribution:
		{
			double d, a, b;

			d = random() * 0.001;
			a = random() * MYPI * 2;
			b = random() * MYPI;

			*x = (0.45 + d) * sin(b) * cos(a);
			*y = (0.45 + d) * sin(b) * sin(a);
			*z = (0.45 + d) * cos(b);

			*x += 0.5;
			*y += 0.5;
			*z += 0.5;
			*x = *x * (max - min) + min;
			*y = *y * (max - min) + min;
			*z = *z * (max - min) + min;
		}
		break;
		default:
		{
			// unknown distribution
			assert(0);
		}
	}
}

// Generate random point set

bool generatePoints(int numofpoint, REAL ** list, Distribution dist)
{
	// Check input
	if (numofpoint <= 0 || list == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}

	// Initialize memory
	*list = new REAL[numofpoint*3];
	if (*list == NULL)
	{
		printf("No more memory\n");
		return false; // no more memory
	}

	// Generate random point set
	std::set<tetVertex, CompareByPosition> pointset;
	int i = 0, tryNum = 0;
	while (i < numofpoint)
	{
		// generate a random point
		REAL x, y, z;
		generateSinglePoint(&x, &y, &z, dist);
		tryNum++;

		// duplication checking
		tetVertex vert(x, y, z);
		if (pointset.find(vert) == pointset.end())
		{
			// no duplicate point, insert this point
			std::pair<std::set<tetVertex>::iterator, bool> ret;
			ret = pointset.insert(vert);
			assert(ret.second);
			(*list)[3 * i] = x;
			(*list)[3 * i + 1] = y;
			(*list)[3 * i + 2] = z;
			i++;
			tryNum = 0;
			// show progress bar
			if (i % ((numofpoint + 9) / 10) == 0)
			{
				int percent = 10 * i / ((numofpoint + 9) / 10);
				printf("%d%% ", percent);
			}
		}
		else
		{
			//if (tryNum > maxtrynum)
			//{
			//	printf("Tried too many times\n");
			//	return false; // exceed maximum try times
			//}
		}
	}
	printf("\n");
	return true;
}

// Generate random triangle set as input constraint face

bool generateTriangles(
	int numofpoint, REAL * pointlist,
	int numofhull, int * hullfaces,
	int numoftri, int ** trilist, double minarea)
{
	// check input
	if (numofpoint <= 0 || pointlist == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}
	if (numofhull > 0 && hullfaces == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}
	if (numoftri < 0 || (numoftri > 0 && trilist == NULL))
	{
		printf("Invalid input\n");
		return false; // invalid input
	}

	// Initialize memory
	*trilist = new int[(numoftri+numofhull) * 3];
	if (*trilist == NULL)
	{
		printf("No more memory\n");
		return false; // no more memory
	}

	// Generate random triangle set
	std::set<tetTri, CompareByThreeEndPoints> triset;
	std::set<tetTri, CompareByThreeEndPoints> failset;
	int i = 0, tryNum = 0;
	while (i < numoftri+numofhull)
	{
		// generate three end points
		int p[3];
		if (i < numofhull)
		{
			for (int j = 0; j < 3; j++)
				p[j] = hullfaces[3 * i + j];
			assert(!(p[0] == p[1] || p[0] == p[2] || p[1] == p[2]));
		}
		else
		{
			for (int j = 0; j < 3; j++)
				p[j] = random()*(numofpoint - 1);
			if (p[0] == p[1] || p[0] == p[2] || p[1] == p[2])
				continue; // duplicated end points
		}

		// Sort
		for (int j = 0; j < 3; j++) // selection sort
		{
			for (int k = j + 1; k < 3; k++)
			{
				if (p[j] > p[k])
				{
					// swap
					int tmp;
					tmp = p[j];
					p[j] = p[k];
					p[k] = tmp;
				}
			}
		}

		tetTri tri(p[0], p[1], p[2]);

		// duplication checking
		if (i < numofhull)
			assert(triset.find(tri) == triset.end());
		if (triset.find(tri) != triset.end()) // duplicated triangle
			continue;

		// failure record checking
		//if (failset.find(tri) != failset.end()) // try already
		//	continue;

		// retry number checking
		tryNum++;
		//if (tryNum > maxtrynum) // try too many times
		//{
		//	printf("Try too many times\n");
		//	return false;
		//}

		// get the coordinate for triangles
		REAL ntri[3][3];
		for (int k = 0; k < 3; k++)
		{
			for (int m = 0; m < 3; m++)
			{
				ntri[k][m] = pointlist[3 * p[k] + m];
			}
		}

		// collinear checking: to avoid collinear (nearly) triangle
		// assume all triangles in convex hull are good
		if (i >= numofhull)
		{
			REAL len, l[2], ret;
			len = distance(ntri[0], ntri[2]); // longest edge
			l[0] = distance(ntri[0], ntri[1]);
			l[1] = distance(ntri[1], ntri[2]);
			for (int k = 0; k < 2; k++)
			{
				if (len < l[k])
				{
					REAL tmp;
					tmp = len;
					len = l[k];
					l[k] = tmp;
				}
			}
			ret = (l[0] + l[1] - len) / len;
			if (ret < 1.0e-8) // triangle too small
			{
				//failset.insert(tri); // record this failed triangle
				continue;
			}

			// area checking: to generate large triangles
			REAL halfPeriTri = (len + l[0] + l[1]) / 2;
			REAL edgeMin = minarea*max; // minarea is between 0-1
			REAL halfPeriMin = (3 * edgeMin) / 2;
			REAL areaTri = halfPeriTri*(halfPeriTri - len)*(halfPeriTri - l[0])*(halfPeriTri - l[1]);
			REAL areaMin = halfPeriMin*(halfPeriMin - edgeMin)*(halfPeriMin - edgeMin)*(halfPeriMin - edgeMin);
			//printf("areaTri = %lf, areaMin = %lf\n", areaTri, areaMin);
			if (areaTri < areaMin) // triangle too small
			{
				//failset.insert(tri); // record this failed triangle
				continue;
			}
		}

		// triangle-triangle intersection checking
		bool flag = false;
		for (int j = 0; j < i; j++)
		{
			// coordinate for new and old triangles
			REAL otri[3][3];
			for (int k = 0; k < 3; k++)
			{
				for (int m = 0; m < 3; m++)
				{
					otri[k][m] = pointlist[3 * (*trilist)[3 * j + k] + m];
				}
			}

			// intersection test
			enum interresult intersect
				= (enum interresult) tri_tri_inter(
					ntri[0], ntri[1], ntri[2],
					otri[0], otri[1], otri[2]);
			assert(intersect != SHAREFACE); // this should not happen
			if (i < numofhull)
				assert(intersect != INTERSECT);
			if (intersect == INTERSECT)
			{
				//printf("Triangle %d and %d intersect each other!\n", i, j);
				//failset.insert(tri); // record this failed triangle
				flag = true;
				break;
			}
		}
		if (!flag) // do not intersect any triangle
		{
			// insert this triangle
			std::pair<std::set<tetTri>::iterator, bool> ret;
			ret = triset.insert(tri);
			assert(ret.second);
			(*trilist)[3 * i] = p[0];
			(*trilist)[3 * i + 1] = p[1];
			(*trilist)[3 * i + 2] = p[2];
			i++;
			tryNum = 0;

			// show progress bar
			if (i >= numofhull && (i - numofhull) % ((numoftri + 9) / 10) == 0)
			{
				int percent = 10 * (i - numofhull) / ((numoftri + 9) / 10);
				printf("%d%% ", percent);
			}
		}
	}
	printf("\n");
	return true;
}

//

// Generate random dangling segment as input segment

bool generateEdges(
	int numofpoint, REAL * pointlist,
	int numoftri, int * trilist,
	int numofedge, int ** edgelist)
{
	// check input
	if (numofpoint <= 0 || pointlist == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}
	if (numoftri > 0 && trilist == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}
	if (numofedge <= 0 || edgelist == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}

	// Initialize memory
	*edgelist = new int[numofedge * 2];
	if (*edgelist == NULL)
		return false; // no more memory

	// Collect existing edge
	std::set<tetEdge, CompareByTwoEndPoints> oldset;
	for (int j = 0; j < numoftri; j++)
	{
		// three endpoints
		int p[3];
		for (int k = 0; k < 3; k++)
		{
			p[k] = trilist[3 * j + k];
		}
		// three edges
		tetEdge e[3];
		for (int k = 0; k < 3; k++)
		{
			e[k].x = p[k];
			e[k].y = p[(k + 1) % 3];
			assert(e[k].x != e[k].y);
			if (e[k].x > e[k].y)
			{
				int tmp = e[k].x;
				e[k].x = e[k].y;
				e[k].y = tmp;
			}
			oldset.insert(e[k]);
		}
	}
	//printf("old edge set size %d\n", oldset.size());

	// Generate random dangling set
	std::set<tetEdge, CompareByTwoEndPoints> edgeset;
	int i = 0, tryNum = 0;
	while (i < numofedge)
	{
		// generate two end points
		int p[2];
		for (int j = 0; j < 2; j++)
			p[j] = random()*(numofpoint - 1);
		if (p[0] == p[1])
			continue; // duplicated end points
		if (p[0] > p[1]) // sort
		{
			// swap
			int tmp;
			tmp = p[0];
			p[0] = p[1];
			p[1] = tmp;
		}
		//printf("New edge %d,%d - ", p[0], p[1]);

		// retry number checking
		tryNum++;
		if (tryNum > maxtrynum) // try too many times
		{
			printf("Try too many times\n");
			return false;
		}

		// duplication checking
		tetEdge e(p[0], p[1]);
		if (edgeset.find(e) != edgeset.end()) // duplicated edge
		{
			//printf("Duplicated edge\n");
			continue;
		}

		// failure record checking
		if (oldset.find(e) != oldset.end()) // try already
		{
			//printf("Tried already\n");
			continue;
		}

		// intersection checking
		bool flag = false;
		REAL nedge[2][3];
		for (int k = 0; k < 2; k++)
		{
			for (int m = 0; m < 3; m++)
			{
				// two endpoints for edge
				nedge[k][m] = pointlist[3 * p[k] + m];
			}
		}

		// triangle - edge intersection checking
		for (int j = 0; j < numoftri; j++)
		{
			// coordinate for old triangles
			REAL otri[3][3];
			for (int k = 0; k < 3; k++)
			{
				for (int m = 0; m < 3; m++)
				{
					otri[k][m] = pointlist[3 * trilist[3 * j + k] + m];
				}
			}

			// intersection test
			int types[2], poss[4];
			int intersect
				= tri_edge_test(
					otri[0], otri[1], otri[2],
					nedge[0], nedge[1],
					NULL, 0, types, poss);
			if (intersect == 1)
			{
				//printf("Intersects with Triangle %d\n", j);
				oldset.insert(e); // record this failed triangle
				flag = true;
				break;
			}
		}

		// edge - edge intersection checking
		for (int j = 0; j < i; j++)
		{
			// endpoint indices for old edges
			int op[2];
			for (int k = 0; k < 2; k++)
			{
				op[k] = (*edgelist)[2 * j + k];
			}

			// coordinate for old edges
			REAL oedge[2][3];
			for (int k = 0; k < 2; k++)
			{
				for (int m = 0; m < 3; m++)
				{
					oedge[k][m] = pointlist[3 * op[k] + m];
				}
			}

			// avoid collinear (nearly) endpoints
			bool cl[2] = { false,false };
			for (int k = 0; k < 2; k++)
			{
				REAL len, l[2], ret;
				len = distance(nedge[0], nedge[1]); // longest edge
				l[0] = distance(nedge[0], oedge[k]);
				l[1] = distance(nedge[1], oedge[k]);
				for (int m = 0; m < 2; m++) // swap
				{
					if (len < l[m])
					{
						REAL tmp;
						tmp = len;
						len = l[m];
						l[m] = tmp;
					}
				}
				ret = (l[0] + l[1] - len) / len;
				if (ret < 1.0e-8) // triangle too small
					cl[k] = true;
			}
			if (cl[0] && cl[1]) // skip when two edges are (nearly) collinear
			{
				flag = true;
				break;
			}

			// if adjacent, dont need to check
			bool aj = false;
			for (int k = 0; k < 2; k++)
			{
				if (p[k] == op[0] || p[k] == op[1])
				{
					aj = true;
					break;
				}
			}
			if (aj)
				continue;

			// intersection test, convert edge-edge to tri-edge
			int types[2], poss[4];
			int intersect
				= tri_edge_test(
					nedge[0], nedge[1], cl[0] ? oedge[1] : oedge[0], // two different combinations
					oedge[0], oedge[1],
					NULL, 1, types, poss);
			if (intersect == 4) // coplanar
			{
				if (types[1] != DISJOINT) // intersect
				{
					oldset.insert(e); // record this failed edge
					flag = true;
					break;
				}
			}
		}

		// do not intersect any triangle or edge
		if (!flag) 
		{
			//printf("Inserted\n");
			std::pair<std::set<tetEdge>::iterator, bool> ret;
			ret = edgeset.insert(e);
			assert(ret.second);
			(*edgelist)[2 * i] = p[0];
			(*edgelist)[2 * i + 1] = p[1];
			i++;
			tryNum = 0;

			// show progress bar
			if (numofedge > 100) // may take a bit long
			{
				if (i % ((numofedge + 10) / 10) == 0)
				{
					int percent = 10 * i / ((numofedge + 10) / 10);
					printf("%d%% ", percent);
				}
			}
		}
	}
	printf("100%%\n");
	return true;
}

// Generate input sets, including point set, triangle set and dangling segment set

bool generateInputSets(
	int numofpoint, REAL ** pointlist,
	int* numoftri, int ** trilist,
	int numofedge, int ** edgelist,
	int seed, Distribution dist, double minarea)
{
	printf("Generating input sets....\n");

	// Initialize random number generator
	randinit(seed, 0, 1000);

	// Declare control variable
	bool flag = true;

	// Generate input point set (compulsory)
	printf("1. Generating input point set(numofpoints = %d)....\n", numofpoint);
	flag = generatePoints(numofpoint, pointlist, dist);
	if (!flag)
		return false;

	// Generate convex hull for point set (compulsory)
	printf("2. Generating convex hull for point set\n");
	tetgenio in, out;
	in.numberofpoints = numofpoint;
	in.pointlist = new REAL[3 * numofpoint];
	for (int i = 0; i < 3*numofpoint; i++)
	{
		in.pointlist[i] = (*pointlist)[i];
	}
	tetrahedralize("Q", &in, &out);
	int numofhull = out.numberoftrifaces;
	int * hullfaces = new int[3 * numofhull];
	for (int i = 0; i < 3*numofhull; i++)
	{
		hullfaces[i] = out.trifacelist[i];
	}
	printf("Hull size = %d\n", numofhull);

	// Generate input triangle set
	printf("3. Generating input triangle set(numoftriangles = %d)....\n", *numoftri);
	flag = generateTriangles(numofpoint, *pointlist, numofhull, hullfaces, *numoftri, trilist, minarea);
	delete[] hullfaces;
	if (!flag)
		return false;
	*numoftri += numofhull;

	if (numofedge > 0)
	{
		// Generate input dangling segment set
		printf("4. Generating input edge set(numofedges = %d)....\n", numofedge);
		flag = generateEdges(numofpoint, *pointlist, *numoftri, *trilist, numofedge, edgelist);
		if (!flag)
			return false;
	}

	return true;
}

// Generate input PLC file

void generateInputPLCFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea)
{
	int numofsubfaces = numoftri; // will be added to hull size
	REAL * pointlist = (REAL*)NULL;
	int * trilist = (int*)NULL;
	int * edgelist = (int*)NULL;
	bool flag =
		generateInputSets(
			numofpoint, &pointlist,
			&numofsubfaces, &trilist,
			numofedge, &edgelist,
			seed, dist, minarea);
	if (!flag)
	{
		printf("Failed to generate input sets\n");
		exit(0);
	}
	// Load input sets to TetGen data structure
	tetgenio in;
	tetgenio::facet * f;
	tetgenio::polygon * p;
	// Point list
	in.firstnumber = 0;
	in.numberofpointattributes = 0;
	in.numberofpoints = numofpoint;
	in.pointlist = pointlist;
	// Facet list, including constraint triangle and edge
	in.numberoffacets = numofsubfaces + numofedge;
	in.facetlist = new tetgenio::facet[in.numberoffacets];
	// Add constraint triangle
	for (int i = 0; i < numofsubfaces; i++)
	{
		f = &in.facetlist[i];
		f->numberofpolygons = 1;
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		p = &f->polygonlist[0];
		p->numberofvertices = 3;
		p->vertexlist = new int[p->numberofvertices];
		for (int k = 0; k < 3; k++)
		{
			p->vertexlist[k] = trilist[3 * i + k];
		}
	}
	// Add constraint edge
	for (int i = 0; i < numofedge; i++)
	{
		f = &in.facetlist[i + numofsubfaces];
		f->numberofpolygons = 1;
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		p = &f->polygonlist[0];
		p->numberofvertices = 2;
		p->vertexlist = new int[p->numberofvertices];
		for (int k = 0; k < 2; k++)
		{
			p->vertexlist[k] = edgelist[2 * i + k];
		}
	}

	// Prepare for filename
	std::ostringstream strs;
	strs << "input_plc/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	// Save PLC to files
	in.save_nodes(com);
	in.save_poly(com);

	// Release memory (some bugs here)
	//in.deinitialize();
	//if (pointlist != (REAL*)NULL)
	//	delete[] pointlist;
	if (trilist != (int*)NULL)
		delete[] trilist;
	if (edgelist != (int*)NULL)
		delete[] edgelist;
}

// Read input PLC file

bool readInputPLCFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out)
{
	assert(out != NULL);

	// Prepare for filename
	std::ostringstream strs;
	strs << "input_plc/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	// Try to open
	strs << ".node";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	FILE *fp;
	fp = fopen(comtmp, "r");
	if (fp == NULL)
		return false;
	fclose(fp);

	// load into TetGen Object
	out->load_poly(com);
	return true;
}

// Generate CDT File using TetGen

void generateInputCDTFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea)
{
	// Prepare for fileName
	std::ostringstream strs;
	strs << "input_cdt/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	tetgenio in, out;
	// Real from PLC file
	bool flag = readInputPLCFile(numofpoint, numoftri, numofedge, seed, dist, minarea, &in);
	if (!flag)
	{
		printf("Failed to read PLC file!\n");
		return;
	}

	// Calculate CDT
	StopWatchInterface *timer = 0; // timer
	sdkCreateTimer(&timer);
	double cpu_time;
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	tetrahedralize("QpLO0/0", &in, &out); // Supposed to terminate
	sdkStopTimer(&timer);
	cpu_time = sdkGetTimerValue(&timer);

	// Save CDT into files
	out.save_elements(com);
	out.save_nodes(com);
	out.save_faces(com);
	out.save_edges(com);

	// Save information into files
	strs << ".txt";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	FILE * fp;
	fp = fopen(comtmp, "w");
	fprintf(fp, "Number of points = %d\n", out.numberofpoints);
	fprintf(fp, "Number of subfaces = %d\n", out.numberoftrifaces);
	fprintf(fp, "Number of segments = %d\n", out.numberofedges);
	fprintf(fp, "Number of tetrahedra = %d\n", out.numberoftetrahedra);
	fprintf(fp, "Runtime = %lf\n", cpu_time);
	fclose(fp);
}

// Read CDT file
bool readInputCDTFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out)
{
	assert(out != NULL);

	// Prepare for filename
	std::ostringstream strs;
	strs << "input_cdt/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	// Try to open
	strs << ".txt";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	FILE *fp;
	fp = fopen(comtmp, "r");
	if (fp == NULL)
		return false;
	fclose(fp);

	// load into TetGen Object
	out->load_node(com);
	out->load_tet(com);
	out->load_face(com);
	out->load_edge(com);
	return true;
}

// Generate random point set

bool generatePoints(int numofpoint, int* curnumofpoint, REAL * list, Distribution dist)
{
	// Check input
	if (numofpoint <= 0 || list == NULL || numofpoint < *curnumofpoint)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}


	std::set<tetVertex, CompareByPosition> pointset;
	int i = 0, tryNum = 0;
	std::pair<std::set<tetVertex>::iterator, bool> ret;
	REAL x, y, z;

	// Insert existing points
	for (i = 0; i < *curnumofpoint; i++)
	{
		tetVertex vert(list[3 * i + 0], list[3 * i + 1], list[3 * i + 2]);
		ret = pointset.insert(vert);
		assert(ret.second);
	}

	// show progress bar
	{
		int percent = 10 * i / ((numofpoint + 9) / 10);
		printf("%d%% ", percent);
	}

	// Continue to generate random point set
	while (i < numofpoint)
	{
		// generate a random point
		generateSinglePoint(&x, &y, &z, dist);
		tryNum++;

		// duplication checking
		tetVertex vert(x, y, z);
		if (pointset.find(vert) == pointset.end())
		{
			// no duplicate point, insert this point
			ret = pointset.insert(vert);
			assert(ret.second);
			list[3 * i] = x;
			list[3 * i + 1] = y;
			list[3 * i + 2] = z;
			i++;
			tryNum = 0;
			// show progress bar
			if (i % ((numofpoint + 9) / 10) == 0)
			{
				int percent = 10 * i / ((numofpoint + 9) / 10);
				printf("%d%% ", percent);
			}
		}
		else
		{
			//if (tryNum > maxtrynum)
			//{
			//	printf("Tried too many times\n");
			//	return false; // exceed maximum try times
			//}
		}
	}
	*curnumofpoint = i;
	printf("\n");
	return true;
}

void sortIndices(int length, int * indices)
{
	// Sort
	for (int j = 0; j < length; j++) // selection sort
	{
		for (int k = j + 1; k < length; k++)
		{
			if (indices[j] > indices[k])
			{
				// swap
				int tmp;
				tmp = indices[j];
				indices[j] = indices[k];
				indices[k] = tmp;
			}
		}
	}
}

bool generateRectangles(
	int numofpoint, int *curnumofpoint, REAL * pointlist,
	int numoftri, int *curnumofrect, int * trilist, 
	Distribution dist, double minarea)
{
	// check input
	if (numofpoint <= 0 || pointlist == NULL)
	{
		printf("Invalid input\n");
		return false; // invalid input
	}

	if (numoftri < 0 || (numoftri > 0 && trilist == NULL))
	{
		printf("Invalid input\n");
		return false; // invalid input
	}

	std::set<tetVertex, CompareByPosition> pointset;
	std::set<tetRect, CompareByFourEndPoints> rectset;
	std::pair<std::set<tetVertex>::iterator, bool> pret;
	std::pair<std::set<tetRect>::iterator, bool> rret;
	int i = 0, j, k, tryNum = 0;
	REAL x, y, z;
	int p[4];

	// Insert existing points and rectangles first (generally from bounding box)
	for (i = 0; i < *curnumofpoint; i++)
	{
		x = pointlist[3 * i + 0];
		y = pointlist[3 * i + 1];
		z = pointlist[3 * i + 2];
		tetVertex vert(x, y, z);
		pret = pointset.insert(vert);
		assert(pret.second);
	}
	for (i = 0; i < *curnumofrect; i++)
	{
		for (j = 0; j < 4; j++)
			p[j] = trilist[4 * i + j];
		sortIndices(4, p);
		tetRect rect(p[0], p[1], p[2], p[3]);
		rret = rectset.insert(rect);
		assert(rret.second);
	}

	// show progress bar
	{
		int percent = 10 * (*curnumofrect) / ((numoftri + 9) / 10);
		printf("%d%% ", percent);
	}

	// Generate random points and rectangles
	// one rectangle contains 4 new points
	while (*curnumofrect < numoftri)
	{
		if (*curnumofpoint + 4 > numofpoint)
		{
			printf("The number of points exceeds the limit because of too many rectangles!");
			return false;
		}

		std::set<tetVertex, CompareByPosition> rectptset;
		REAL rectpt[4][3];
		REAL vector[4][3], t;
		REAL shortlength = 0.001*max; // minimum edge length
		REAL areamin = minarea*minarea*max*max; // minium area
		REAL length, height;
		bool flag = false;
		i = 0;

		// Generate three random points to define a plane where the rectangle is located
		while (i < 3)
		{
			generateSinglePoint(&x, &y, &z, dist);
			tetVertex vert(x, y, z);
			if (rectptset.find(vert) == rectptset.end() &&
				pointset.find(vert) == pointset.end()) // not a duplicate point
			{
				if (i == 0) // always insert the first point
				{
					rectpt[0][0] = x;
					rectpt[0][1] = y;
					rectpt[0][2] = z;
					pret = rectptset.insert(vert);
					assert(pret.second);
					flag = true;
				}
				else if (i == 1) // avoid to form a short edge
				{
					rectpt[1][0] = x;
					rectpt[1][1] = y;
					rectpt[1][2] = z;
					length = distance(rectpt[0], rectpt[1]);
					if (length >= shortlength)
					{
						pret = rectptset.insert(vert);
						assert(pret.second);
						flag = true;
					}
				}
				else if (i == 2) // avoid to form a small distance to the edge
				{
					rectpt[2][0] = x;
					rectpt[2][1] = y;
					rectpt[2][2] = z;
					vector[0][0] = (rectpt[0][0] - rectpt[1][0]) / length; // identity vector from point 1 to point 0
					vector[0][1] = (rectpt[0][1] - rectpt[1][1]) / length;
					vector[0][2] = (rectpt[0][2] - rectpt[1][2]) / length;
					vector[1][0] = rectpt[2][0] - rectpt[1][0]; // vector from point 1 to point 2
					vector[1][1] = rectpt[2][1] - rectpt[1][1];
					vector[1][2] = rectpt[2][2] - rectpt[1][2];
					t = vector[0][0] * vector[1][0] + vector[0][1] * vector[1][1] + vector[0][2] * vector[1][2];
					vector[2][0] = rectpt[1][0] + t*vector[0][0]; // projected point
					vector[2][1] = rectpt[1][1] + t*vector[0][1];
					vector[2][2] = rectpt[1][2] + t*vector[0][2];
					vector[3][0] = rectpt[2][0] - vector[2][0]; // vector from pprojected point to point 2
					vector[3][1] = rectpt[2][1] - vector[2][1];
					vector[3][2] = rectpt[2][2] - vector[2][2];
					
					height = sqrt(vector[3][0] * vector[3][0] + vector[3][1] * vector[3][1] + vector[3][2] * vector[3][2]);
					if (height >= shortlength && length*height >= areamin)
					{
						pret = rectptset.insert(vert);
						assert(pret.second);
						flag = true;
					}
				}
			}
			else
				flag = false;
			if (flag)
			{
				i++;
			}
		}

		flag = true; // reject a rectangle when this becomes false
		// Compute point 2 and point 3 as rectangle's endpoints
		rectpt[2][0] = rectpt[1][0] + vector[3][0];
		rectpt[2][1] = rectpt[1][1] + vector[3][1];
		rectpt[2][2] = rectpt[1][2] + vector[3][2];
		rectpt[3][0] = rectpt[0][0] + vector[3][0];
		rectpt[3][1] = rectpt[0][1] + vector[3][1];
		rectpt[3][2] = rectpt[0][2] + vector[3][2];

		for (i = 0; i < 2; i++)
		{
			for (j = 0; j < 3; j++)
			{
				if (rectpt[2 + i][j] <= min || rectpt[2 + i][j] >= max)
				{
					flag = false;
					break;
				}
			}
			if (!flag)
				break;
			tetVertex vert(rectpt[2 + i][0], rectpt[2 + i][1], rectpt[2 + i][2]);
			if (pointset.find(vert) != pointset.end()) // duplicate endpoints
			{
				flag = false;
				break;
			}
		}

		if (!flag)
			continue;

		// rectangle-rectangle intersection checking
		for (i = 0; i < *curnumofrect; i++)
		{
			for (j = 0; j < 4; j++)
			{
				int p = trilist[4 * i + j];
				for (k = 0; k < 3; k++)
				{
					vector[j][k] = pointlist[3 * p + k];
				}
			}

			if (rect_rect_inter(
				rectpt[0], rectpt[1], rectpt[2], rectpt[3],
				vector[0], vector[1], vector[2], vector[3]))
			{
				flag = false;
				break;
			}
		}

		if (!flag)
			continue;

		// This rectangle is ready to insert now
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 3; j++)
			{
				pointlist[3 * (*curnumofpoint + i) + j] = rectpt[i][j];
			}
			tetVertex vert(rectpt[i][0], rectpt[i][1], rectpt[i][2]);
			pret = pointset.insert(vert);
			assert(pret.second);
		}

		tetRect rect(*curnumofpoint, *curnumofpoint + 1, *curnumofpoint + 2, *curnumofpoint + 3);
		rret = rectset.insert(rect);
		assert(rret.second);
		for (i = 0; i < 4; i++)
		{
			trilist[4 * (*curnumofrect) + i] = *curnumofpoint + i;
		}

		// Update variables
		*curnumofpoint += 4;
		*curnumofrect += 1;

		// show progress bar
		if ( *curnumofrect % ((numoftri + 9) / 10) == 0)
		{
			int percent = 10 * (*curnumofrect) / ((numoftri + 9) / 10);
			printf("%d%% ", percent);
		}
	}
	printf("\n");
	return true;
}

bool readOldInputSets_without_acute_angles(
	int numofpoint, REAL* pointlist,
	int numoftri, int* trilist,
	int numofedge, int *edgelist,
	int & curnumofpoint, int& curnumofrect,
	int seed, Distribution dist, double minarea)
{
	int i = numoftri - 100, j;
	tetgenio out;
	bool flag = false;
	while (i >= 0)
	{
		flag = readInputPLCFile_without_acute_angles(numofpoint, i, numofedge, seed, dist, minarea, &out);
		if (flag)
			break;
		i -= 100;
	}
	if (flag)
	{
		curnumofpoint = (i - 6) * 4 + 8;
		curnumofrect = i;
		for (i = 0; i < curnumofpoint; i++)
		{
			for (j = 0; j < 3; j++)
			{
				pointlist[3 * i + j] = out.pointlist[3 * i + j];
			}
		}
		tetgenio::facet * f;
		tetgenio::polygon * p;
		for (i = 0; i < curnumofrect; i++)
		{
			f = &out.facetlist[i];
			assert(f->numberofpolygons == 1);
			p = &f->polygonlist[0];
			assert(p->numberofvertices == 4);
			for (j = 0; j < 4; j++)
			{
				trilist[4 * i + j] = p->vertexlist[j];
			}
		}
	}
	return flag;
}

// Generate input sets, including point set, triangle set and dangling segment set
// without any acute plc angles (dihedral, segment-facet and segment-segment angles)

bool generateInputSets_without_acute_angles(
	int* numofpoint, REAL ** pointlist,
	int* numoftri, int ** trilist,
	int* numofedge, int ** edgelist,
	int seed, Distribution dist, double minarea)
{
	printf("Generating input sets without acute angles....\n");

	// Initialize random number generator
	randinit(seed, 0, 1000);

	// Declare control variable
	bool flag = true;

	// Input checking
	if (*numofpoint < 8) // reserve space for bounding box
	{
		printf("The number of points must not be less than 8.\n");
		return false;
	}

	if (*numoftri < 6) // reseve space for bounding box
	{
		printf("The number of subfaces must not be less than 6.\n");
		return false;
	}

	*pointlist = new REAL[3 * (*numofpoint)];
	*trilist = new int[4 * (*numoftri)];
	int curnumofpoint = 8;
	int curnumofrect = 6;

	//if (readOldInputSets_without_acute_angles(
	//	*numofpoint, *pointlist,
	//	*numoftri, *trilist,
	//	*numofedge, *edgelist,
	//	curnumofpoint, curnumofrect,
	//	seed, dist, minarea))
	//{
	//	printf("1. Read existing PLC file to speed up the process.... Succeeded\n");
	//}
	//else
	{
		curnumofpoint = 8;
		curnumofrect = 6;
		printf("1. Generating bounding box....\n");
		(*pointlist)[0] = 0;
		(*pointlist)[1] = 0;
		(*pointlist)[2] = 0; // 0
		(*pointlist)[3] = 0;
		(*pointlist)[4] = 1000;
		(*pointlist)[5] = 0; // 1
		(*pointlist)[6] = 1000;
		(*pointlist)[7] = 1000;
		(*pointlist)[8] = 0; // 2
		(*pointlist)[9] = 1000;
		(*pointlist)[10] = 0;
		(*pointlist)[11] = 0;// 3
		(*pointlist)[12] = 0;
		(*pointlist)[13] = 0;
		(*pointlist)[14] = 1000; // 4
		(*pointlist)[15] = 0;
		(*pointlist)[16] = 1000;
		(*pointlist)[17] = 1000; // 5
		(*pointlist)[18] = 1000;
		(*pointlist)[19] = 1000;
		(*pointlist)[20] = 1000; // 6
		(*pointlist)[21] = 1000;
		(*pointlist)[22] = 0;
		(*pointlist)[23] = 1000; // 7

		(*trilist)[0] = 0;
		(*trilist)[1] = 1;
		(*trilist)[2] = 2;
		(*trilist)[3] = 3; // 0
		(*trilist)[4] = 4;
		(*trilist)[5] = 5;
		(*trilist)[6] = 6;
		(*trilist)[7] = 7; // 1
		(*trilist)[8] = 0;
		(*trilist)[9] = 1;
		(*trilist)[10] = 5;
		(*trilist)[11] = 4;// 2
		(*trilist)[12] = 2;
		(*trilist)[13] = 3;
		(*trilist)[14] = 7;
		(*trilist)[15] = 6;// 3
		(*trilist)[16] = 0;
		(*trilist)[17] = 4;
		(*trilist)[18] = 7;
		(*trilist)[19] = 3;// 4
		(*trilist)[20] = 1;
		(*trilist)[21] = 2;
		(*trilist)[22] = 6;
		(*trilist)[23] = 5;// 5
	}

	printf("2. Generating input rectangle set (numofrectangles = %d)....\n", *numoftri);
	flag = generateRectangles(
		*numofpoint, &curnumofpoint, *pointlist, 
		*numoftri, &curnumofrect, *trilist, 
		dist, minarea);
	if (!flag)
		return false;

	printf("3. Generating input point set (numofpoints = %d)....\n", *numofpoint);
	flag = generatePoints(*numofpoint, &curnumofpoint, *pointlist, dist);
	if (!flag)
		return false;

	//if (numofedge > 0)
	//{
	//	// Generate input dangling segment set
	//	printf("4. Generating input edge set(numofedges = %d)....\n", numofedge);
	//	flag = generateEdges(numofpoint, *pointlist, *numoftri, *trilist, numofedge, edgelist);
	//	if (!flag)
	//		return false;
	//}

	return true;
}

// Generate input PLC file

void generateInputPLCFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea)
{
	int numofsubfaces = numoftri; // will be added to hull size
	REAL * pointlist = (REAL*)NULL;
	int * trilist = (int*)NULL;
	int * edgelist = (int*)NULL;
	bool flag =
		generateInputSets_without_acute_angles(
			&numofpoint, &pointlist,
			&numofsubfaces, &trilist,
			&numofedge, &edgelist,
			seed, dist, minarea);
	if (!flag)
	{
		printf("Failed to generate input sets without acute angles\n");
		exit(0);
	}

	// Load input sets to TetGen data structure
	tetgenio in;
	tetgenio::facet * f;
	tetgenio::polygon * p;
	// Point list
	in.firstnumber = 0;
	in.numberofpointattributes = 0;
	in.numberofpoints = numofpoint;
	in.pointlist = pointlist;
	// Facet list, including constraint triangle and edge
	in.numberoffacets = numofsubfaces + numofedge;
	in.facetlist = new tetgenio::facet[in.numberoffacets];
	// Add constraint rectangles
	for (int i = 0; i < numofsubfaces; i++)
	{
		f = &in.facetlist[i];
		f->numberofpolygons = 1;
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		p = &f->polygonlist[0];
		p->numberofvertices = 4;
		p->vertexlist = new int[p->numberofvertices];
		for (int k = 0; k < 4; k++)
		{
			p->vertexlist[k] = trilist[4 * i + k];
		}
	}
	// Add constraint edge
	for (int i = 0; i < numofedge; i++)
	{
		f = &in.facetlist[i + numofsubfaces];
		f->numberofpolygons = 1;
		f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
		f->numberofholes = 0;
		f->holelist = NULL;
		p = &f->polygonlist[0];
		p->numberofvertices = 2;
		p->vertexlist = new int[p->numberofvertices];
		for (int k = 0; k < 2; k++)
		{
			p->vertexlist[k] = edgelist[2 * i + k];
		}
	}

	// Prepare for filename
	std::ostringstream strs;
	strs << "input_plc_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	// Save PLC to files
	in.save_nodes(com);
	in.save_poly(com);

	// Release memory (some bugs here)
	//in.deinitialize();
	//if (pointlist != (REAL*)NULL)
	//	delete[] pointlist;
	if (trilist != (int*)NULL)
		delete[] trilist;
	if (edgelist != (int*)NULL)
		delete[] edgelist;
}

// Read input PLC file

bool readInputPLCFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out)
{
	assert(out != NULL);

	// Prepare for filename
	std::ostringstream strs;
	strs << "input_plc_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	// Try to open
	strs << ".node";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	FILE *fp;
	fp = fopen(comtmp, "r");
	if (fp == NULL)
		return false;
	fclose(fp);

	// load into TetGen Object
	out->load_poly(com);
	return true;
}

// Generate CDT File using TetGen

void generateInputCDTFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea)
{
	// Prepare for fileName
	std::ostringstream strs;
	strs << "input_cdt_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	tetgenio in, out;
	// Real from PLC file
	bool flag = readInputPLCFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &in);
	if (!flag)
	{
		printf("Failed to read PLC file!\n");
		return;
	}

	// Calculate CDT
	StopWatchInterface *timer = 0; // timer
	sdkCreateTimer(&timer);
	double cpu_time;
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	tetrahedralize("VQpLO0/0", &in, &out); // Supposed to terminate
	sdkStopTimer(&timer);
	cpu_time = sdkGetTimerValue(&timer);

	// Save CDT into files
	out.save_elements(com);
	out.save_nodes(com);
	out.save_faces(com);
	out.save_edges(com);

	// Save information into files
	strs << ".txt";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	FILE * fp;
	fp = fopen(comtmp, "w");
	fprintf(fp, "Number of points = %d\n", out.numberofpoints);
	fprintf(fp, "Number of subfaces = %d\n", out.numberoftrifaces);
	fprintf(fp, "Number of segments = %d\n", out.numberofedges);
	fprintf(fp, "Number of tetrahedra = %d\n", out.numberoftetrahedra);
	fprintf(fp, "Runtime = %lf\n", cpu_time);
	fclose(fp);
}

// Read CDT file
bool readInputCDTFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out)
{
	assert(out != NULL);

	// Prepare for filename
	std::ostringstream strs;
	strs << "input_cdt_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	// Try to open
	strs << ".txt";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	FILE *fp;
	fp = fopen(comtmp, "r");
	if (fp == NULL)
		return false;
	fclose(fp);

	// load into TetGen Object
	out->load_node(com);
	out->load_tet(com);
	out->load_face(com);
	out->load_edge(com);
	return true;
}

// Prepare for GPU input
bool prepareInputCDT_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, 
	Distribution dist, 
	double minarea,
	double ratio,
	int mode,
	tetgenio * out)
{
	tetgenio in;
	// Real from PLC file
	bool flag = readInputPLCFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &in);
	if (!flag)
	{
		printf("Failed to read PLC file!\n");
		return false;
	}

	// Calculate CDT
	char *com;
	std::ostringstream strs;
	if (mode == 1)
		strs << "QpnLO0/0";
	else if(mode == 2)
		strs << "QGpnnLq" << ratio << "D2O0/0";
	std::string fn = strs.str();
	com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	tetrahedralize(com, &in, out); // Supposed to terminate

	return true;
}

// Generate off files for CGAL
void generateInputOFFFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea, tetgenio *out)
{
	// Prepare for fileName
	std::ostringstream strs;
	strs << "input_off_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge << "_2.off";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	// Save information into files
	FILE * fp;
	fp = fopen(comtmp, "w");
	fprintf(fp, "OFF\n");
	fprintf(fp, "%d %d %d\n", out->numberofpoints, 2*out->numberoffacets, 0);
	fprintf(fp, "\n");
	for (int i = 0; i < out->numberofpoints; i++)
	{
		fprintf(fp, "%lf %lf %lf\n", out->pointlist[3 * i], out->pointlist[3 * i + 1], out->pointlist[3 * i + 2]);
	}

	tetgenio::facet * f;
	tetgenio::polygon * p;
	for (int i = 0; i < out->numberoffacets; i++)
	{
		f = &out->facetlist[i];
		p = &f->polygonlist[0];
		fprintf(fp, "%d  %d %d %d\n", 3,
			p->vertexlist[0], p->vertexlist[1], p->vertexlist[2]);
		fprintf(fp, "%d  %d %d %d\n", 3,
			p->vertexlist[0], p->vertexlist[2], p->vertexlist[3]);
	}
	fclose(fp);
}

void generateInputOFFFile_without_acute_angles2(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea, tetgenio *out)
{
	// Prepare for fileName
	std::ostringstream strs;
	strs << "input_off_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge << ".off";
	std::string fntmp = strs.str();
	char *comtmp = new char[fntmp.length() + 1];
	strcpy(comtmp, fntmp.c_str());

	// Save information into files
	FILE * fp;
	fp = fopen(comtmp, "w");
	fprintf(fp, "OFF\n");
	fprintf(fp, "%d %d %d\n", 4 * (out->numberoffacets - 6), 2 * (out->numberoffacets - 6), 0);
	fprintf(fp, "\n");
	for (int i = 8; i < 4 * (out->numberoffacets - 6) + 8; i++)
	{
		fprintf(fp, "%lf %lf %lf\n", out->pointlist[3 * i], out->pointlist[3 * i + 1], out->pointlist[3 * i + 2]);
	}

	tetgenio::facet * f;
	tetgenio::polygon * p;
	for (int i = 6; i < out->numberoffacets; i++)
	{
		f = &out->facetlist[i];
		p = &f->polygonlist[0];
		fprintf(fp, "%d  %d %d %d\n", 3,
			p->vertexlist[0] - 8, p->vertexlist[1] - 8, p->vertexlist[2] - 8);
		fprintf(fp, "%d  %d %d %d\n", 3,
			p->vertexlist[0] - 8, p->vertexlist[2] - 8, p->vertexlist[3] - 8);
	}
	fclose(fp);
}