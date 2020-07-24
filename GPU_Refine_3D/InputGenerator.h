#pragma once

#include <math.h>
#include <vector>
#include <set>
#include "tetgen.h"

// Rename type
#define REAL double

// Constant number
#define MYPI 3.141592653589793238462643383279502884197169399375105820974944592308

// enumeration type
enum Distribution
{
	UniformDistribution,
	GaussianDistribution,
	BallDistribution,
	SphereDistribution,
	GridDistribution,
	ThinSphereDistribution
};

// Some useful structures
typedef struct vertex
{
	REAL x, y, z;
	vertex(void) {}
	vertex(REAL a, REAL b, REAL c) { x = a; y = b; z = c; }

} tetVertex;

struct CompareByPosition {
	bool operator()(const tetVertex &lhs, const tetVertex &rhs) const {
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		else if (lhs.y != rhs.y)
			return lhs.y < rhs.y;
		return lhs.z < rhs.z;
	}
};

typedef struct edge
{
	int x, y;
	edge(void) {}
	edge(int a, int b) { x = a; y = b; }
} tetEdge;

struct CompareByTwoEndPoints {
	bool operator()(const tetEdge &lhs, const tetEdge &rhs) const {
		// assume two end points are sorted
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		return lhs.y < rhs.y;
	}
};

typedef struct triangle
{
	int x, y, z;
	triangle(void) {};
	triangle(int a, int b, int c) { x = a; y = b; z = c; }
} tetTri;

struct CompareByThreeEndPoints {
	bool operator()(const tetTri &lhs, const tetTri &rhs) const {
		// assume three end points are sorted
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		else if (lhs.y != rhs.y)
			return lhs.y < rhs.y;
		return lhs.z < rhs.z;
	}
};

typedef struct rectangle
{
	int x, y, z, w;
	rectangle(void) {};
	rectangle(int a, int b, int c, int d) { x = a; y = b; z = c; w = d; }
} tetRect;

struct CompareByFourEndPoints {
	bool operator()(const tetRect &lhs, const tetRect &rhs) const {
		// assume four end points are sorted
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		else if (lhs.y != rhs.y)
			return lhs.y < rhs.y;
		else if (lhs.z != rhs.z)
			return lhs.z < rhs.z;
		return lhs.w < rhs.w;
	}
};

void generateInputPLCFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea);

bool readInputPLCFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out);

void generateInputCDTFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea);

bool readInputCDTFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out);

void generateInputPLCFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea);

bool readInputPLCFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out);

void generateInputCDTFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed, Distribution dist, double minarea);

bool readInputCDTFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio * out);

bool prepareInputCDT_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	double ratio,
	int mode,
	tetgenio * out);

void generateInputOFFFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio *out);

void generateInputOFFFile_without_acute_angles2(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	tetgenio *out);