#pragma once

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
);

void makesegment2parentmap(
	int numofsegment,
	int* segmentlist,
	trihandle* seg2trilist,
	int*& segment2parentlist,
	int*& segmentendpointslist,
	int& numofparent
);

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
);