#pragma once

void saveTr2Mesh(
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist,
	int numoftet,
	int* tetlist,
	char* filename
);

void cutMesh(
	char* infile,
	char* outfile,
	double* plane
);

void convertPLY2Tr(
	char* filename,
	int header_length,
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist
);

void convertPLY2Mesh(
	char* infile,
	int header_length,
	int numofpoint,
	int numoftriface,
	char* outfile
);

void convertOFF2Mesh(
	char* infile,
	char* outfile
);

void convertPLC2Mesh(
	char* infile,
	char* outfile
);

void fillOFF(
	char* infile,
	char* outfile
);

void convertPLY2PLC(
	char* infile,
	int header_length,
	int numofpoint,
	int numoftriface,
	char* outfile1,
	char* outfile2,
	bool duplicated
);

void convertOFF2PLC(
	char* infile,
	char* outfile1,
	char* outfile2
);

void computeCDTofInputFile(
	char* infile,
	char* outfile,
	int format // 0:PLC, 1:OFF
);

void refineInputFileOnCPU(
	char* infile,
	int format,
	double B,
	char* outmesh,
	char* outdata
);

void refineInputFileOnGPU(
	char* infile,
	MESHBH* input_behavior,
	char* outmesh,
	char* outdata
);