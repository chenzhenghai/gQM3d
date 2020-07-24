#pragma once

#include <cuda_runtime.h>

typedef struct MESHIO
{
	int numofpoints;
	double* pointlist;
	int numofedges;
	int* edgelist;
	int numoftrifaces;
	int* trifacelist;
	int numoftets;
	int* tetlist;

	int interneigh;
	double* interpointradius;
	int* intertri2tetlist;
	int* intertri2trilist;
	int* intertet2tetlist;

	MESHIO(void)
	{
		numofpoints = numofedges = numoftets = 0;
		pointlist = NULL;
		edgelist = NULL;
		trifacelist = NULL;
		tetlist = NULL;

		interneigh = 0;
		interpointradius = NULL;
		intertri2tetlist = NULL;
		intertri2trilist = NULL;
		intertet2tetlist = NULL;
	}
} MESHIO;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for tetrahedron													 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct tethandle
{
	int id; // index for tetrahedron in tetrahedron list
	char ver; // Range from 0 to 11.
			  //           |   edge 0     edge 1     edge 2    //
			  //   --------|--------------------------------   //
			  //    face 0 |   0 (0/1)    4 (0/3)    8 (0/2)   //
			  //    face 1 |   1 (1/2)    5 (1/3)    9 (1/0)   //
			  //    face 2 |   2 (2/3)    6 (2/1)   10 (2/0)   //
			  //    face 3 |   3 (3/0)    7 (3/1)   11 (3/2)   //
	__forceinline__ __host__ __device__
	tethandle(void)
	{
		this->id = -1;
		ver = 11;
	}

	__forceinline__ __host__ __device__
	tethandle(int id, char ver)
	{
		this->id = id;
		this->ver = ver;
	}

} tethandle;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for triangle														 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct trihandle
{
	int id; // index for triangle in triangle list
	char shver; // Range from 0 to 5.
				//                 | edge 0   edge 1   edge 2     //
				//  ---------------|--------------------------    //
				//   ccw orieation |   0        2        4        //
				//    cw orieation |   1        3        5		  //
	
	__forceinline__ __host__ __device__
	trihandle(void)
	{
		this->id = -1;
		shver = 0;
	}

	__forceinline__ __host__ __device__
	trihandle(int id, char ver)
	{
		this->id = id;
		this->shver = ver;
	}

} trihandle;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for tetrahedron status											 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct tetstatus
{
	unsigned char _status;
	// 76543210
	// ||||||||----- tetrahedron status
	// |||||||------ abort status
	// ||||||------- bad quality
	// |||||-------- cavity reuse
	// ||||--------- 

	__forceinline__ __host__ __device__
	tetstatus(void)
	{
		_status = 0;
	}

	__forceinline__ __host__ __device__
	tetstatus(unsigned char s)
	{
		_status = s;
	}

	__forceinline__ __host__ __device__
	void clear()
	{
		_status = 0; // clear all information
	}

	//status of tetrahedron. 0 = empty; 1 = non-empty
	__forceinline__ __host__ __device__
	void setEmpty(bool n)
	{
		if (n)
			_status = _status & (~1);
		else
			_status = _status | 1;
	}

	__forceinline__ __host__ __device__
	bool isEmpty() const
	{
		return (_status & 1) == 0;
	}

	//0 = not abortive; 1 = abortive
	__forceinline__ __host__ __device__
		void setAbortive(bool a)
	{
		_status = (_status & ~(1 << 1)) | (a ? 1 : 0) << 1;
	}

	__forceinline__ __host__ __device__
		bool isAbortive() const
	{
		return (_status & (1 << 1)) > 0;
	}

	__forceinline__ __host__ __device__
	void setBad(bool b)
	{
		_status = (_status & ~(1 << 2)) | (b ? 1 : 0) << 2;
	}

	__forceinline__ __host__ __device__
	bool isBad() const
	{
		return (_status & (1 << 2)) > 0;
	}

	__forceinline__ __host__ __device__
	void setCavityReuse(bool b)
	{
		_status = (_status & ~(1 << 3)) | (b ? 1 : 0) << 3;
	}

	__forceinline__ __host__ __device__
		bool isCavityReuse() const
	{
		return (_status & (1 << 3)) > 0;
	}
} tetstatus;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for subface status											     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct tristatus
{
	unsigned char _status;
	// 76543210
	// ||||||||----- tetrahedron status
	// |||||||------ abort status
	// ||||||------- cavity reuse
	// |||||-------- 
	// ||||--------- 

	__forceinline__ __host__ __device__
	tristatus(void)
	{
		_status = 0;
	}

	__forceinline__ __host__ __device__
	tristatus(unsigned char s)
	{
		_status = s;
	}

	__forceinline__ __host__ __device__
	void clear()
	{
		_status = 0; // clear all information
	}

	//status of triface. 0 = empty; 1 = non-empty
	__forceinline__ __host__ __device__
	void setEmpty(bool n)
	{
		if (n)
			_status = _status & (~1);
		else
			_status = _status | 1;
	}

	__forceinline__ __host__ __device__
	bool isEmpty() const
	{
		return (_status & 1) == 0;
	}

	//0 = not abortive; 1 = abortive
	__forceinline__ __host__ __device__
	void setAbortive(bool a)
	{
		_status = (_status & ~(1 << 1)) | (a ? 1 : 0) << 1;
	}

	__forceinline__ __host__ __device__
	bool isAbortive() const
	{
		return (_status & (1 << 1)) > 0;
	}

	__forceinline__ __host__ __device__
	void setCavityReuse(bool b)
	{
		_status = (_status & ~(1 << 2)) | (b ? 1 : 0) << 2;
	}

	__forceinline__ __host__ __device__
	bool isCavityReuse() const
	{
		return (_status & (1 << 2)) > 0;
	}
} tristatus;

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Labels (enumeration declarations) used by TetGen.                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// Labels that signify the type of a vertex. 
enum verttype {
	UNUSEDVERTEX, DUPLICATEDVERTEX, RIDGEVERTEX, ACUTEVERTEX,
	FACETVERTEX, VOLVERTEX, FREESEGVERTEX, FREEFACETVERTEX,
	FREEVOLVERTEX, NREGULARVERTEX, DEADVERTEX
};

// Labels that signify the result of triangle-triangle intersection test.
enum interresult {
	DISJOINT, INTERSECT, SHAREVERT, SHAREEDGE, SHAREFACE,
	TOUCHEDGE, TOUCHFACE, ACROSSVERT, ACROSSEDGE, ACROSSFACE,
	COLLISIONFACE, ACROSSSEG, ACROSSSUB
};

// Labels that signify the result of point location.
enum locateresult {
	UNKNOWN, OUTSIDE, INTETRAHEDRON, ONFACE, ONEDGE, ONVERTEX,
	ENCVERTEX, ENCSEGMENT, ENCSUBFACE, NEARVERTEX, NONREGULAR,
	INSTAR, BADELEMENT
};

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Handle for visualization													 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
typedef struct internalmesh
{
	// iteration
	int iter_seg;
	int iter_subface;
	int iter_tet;
	// animation
	bool animation;
	int* anicolor;
	int framenum;
	// mesh
	int numofpoints;
	double* pointlist;
	verttype* pointtype;
	int numofsubseg;
	int* seglist;
	tristatus* segstatus;
	int numofsubface;
	int* trifacelist;
	tristatus* tristatus;
	tethandle* tri2tetlist;
	int numoftet;
	int* tetlist;
	tethandle* neighborlist;
	unsigned long long* tetmarker;
	tetstatus* tetstatus;
	tethandle* locatedtet;
	// cavity
	int numofthread;
	int *threadlist;
	int numofinsertpt;
	int insertiontype;
	int* insertidxlist;
	double* insertptlist;
	int* threadmarker;
	tethandle* cavebdrylist;
	int* cavebdrynext;
	int* cavebdryhead;
	tethandle* caveoldtetlist;
	int* caveoldtetnext;
	int* caveoldtethead;

	internalmesh(void)
	{
		iter_seg = -1;
		iter_subface = -1;
		iter_tet = -1;
		animation = false;
		framenum = 0;
	}

	internalmesh(int iseg, int isub, int itet)
	{
		iter_seg = iseg;
		iter_subface = isub;
		iter_tet = itet;
		animation = false;
		framenum = 0;
	}
} internalmesh;

typedef struct MESHBH
{
	double radius_to_edge_ratio;
	int minsplittabletets;
	int minbadtets;
	int miniter;
	int minthread;

	int fileformat;
	
	bool R1, R2, R3, R4, R5;

	int cpumode;
	int vecmode;

	int filtermode;
	int filterstatus;
	int maxbadelements;

	int cavetetsize;
	double cavetetsizefac;
	int caveoldtetsize;
	double caveoldtetsizefac;
	int cavebdrysize;
	double cavebdrysizefac;

	int caveshsize;
	double caveshsizefac;

	int cavitymode;
	int maxcavity;
	int mincavity;

	internalmesh* drawmesh;
	double times[6];

	MESHBH(void)
	{
		// input variables
		radius_to_edge_ratio = 1.6;
		minsplittabletets = 0;
		minbadtets = 0;
		miniter = 0;
		minthread = 0;

		// 0: PLC
		// 1: OFF
		fileformat = 1;

		// Rules
		R1 = R2 = R5 = false;
		R3 = R4 = true;

		// 1: compute CDT on the CPU, then refine on the GPU
		// 2: compute CDT, split encroached subsegments and subfaces on the CPU,
		// then refine tetrahedra on the GPU
		cpumode = 1;

		// 1: refine subsegments, subfaces and tetrahedra
		// 2: refine all bad elements together
		vecmode = 2;

		// 1: normal filtering
		// 2: fast filtering
		filtermode = 1;
		filterstatus = 1; // 1: not on, 2: on, 3: just off
		maxbadelements = 1000000;

		cavetetsize = 10000;
		cavetetsizefac = 1.1;
		caveoldtetsize = 10000;
		caveoldtetsizefac = 1.1;
		cavebdrysize = 10000;
		cavebdrysizefac = 1.1;

		caveshsize = 10000;
		caveshsizefac = 1.1;

		// 1: cut cavity when it exceeds maxcavity
		// 2: record cavity when it exceeds mincavity
		cavitymode = 2;
		maxcavity = 500;
		mincavity = 80;

		drawmesh = NULL;
	}
} MESHBH;