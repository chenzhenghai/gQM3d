#include "InputGenerator.h"
#include "tetgen.h"
#include "Experiment.h"
#include "Experiment_CGAL.h"
#include "MeshChecker.h"
#include "Viewer.h"
#include "CudaRefine.h"
#include "MeshRefine.h"

/**
* Host main routine
*/
int
main(int argc, char** argv)
{
	MESHBH behavior;
	behavior.radius_to_edge_ratio = 1.4;
	behavior.minbadtets = 1670000;
	bool outputmesh = false;
	bool outputrep = false;

	// Synthetic
	if (false)
	{
		int numofpoint = 25000;
		int numoftri = 5000;
		int numofedge = 0;
		Distribution dist = (Distribution)2;
		int seed = 0;
		double minArea = 0;

		char* inputpath = "input/";
		char* outputpath = "result/";

		behavior.fileformat = 0; // synthetic plc file

		char *inputfn, *inputfile, *outputmeshfile, *outputrepfile;

		std::ostringstream strs;
		strs << "d" << dist << "_s" << seed << "_a" << minArea
			<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge;
		inputfn = new char[strs.str().length() + 1];
		strcpy(inputfn, strs.str().c_str());

		std::ostringstream strs1;
		strs1 << inputpath << inputfn;
		inputfile = new char[strs1.str().length() + 1];
		strcpy(inputfile, strs1.str().c_str());

		std::ostringstream strs2;
		strs2 << outputpath << inputfn << "_q" << behavior.radius_to_edge_ratio << "_noflip_gpu_new.mesh";
		outputmeshfile = new char[strs2.str().length() + 1];
		strcpy(outputmeshfile, strs2.str().c_str());

		std::ostringstream strs3;
		strs3 << outputpath << inputfn << "_q" << behavior.radius_to_edge_ratio << "_noflip_gpu_new.txt";
		outputrepfile = new char[strs3.str().length() + 1];
		strcpy(outputrepfile, strs3.str().c_str());

		refineInputFileOnGPU(
			inputfile,
			&behavior,
			outputmesh ? outputmeshfile : NULL,
			outputrep ? outputrepfile : NULL
		);
	}

	// Real-world
	if (true)
	{
		behavior.fileformat = 1; // real-world off file

		char* inputpath = "input_real/";
		char* outputpath = "result_real/";

		char* inputfn = "skull";
		char *inputfile, *outputmeshfile, *outputrepfile;

		std::ostringstream strs;
		strs << inputpath << inputfn << ".off";
		inputfile = new char[strs.str().length() + 1];
		strcpy(inputfile, strs.str().c_str());

		std::ostringstream strs1;
		strs1 << outputpath << inputfn << "_gpu_new_" << behavior.radius_to_edge_ratio << ".mesh";
		outputmeshfile = new char[strs1.str().length() + 1];
		strcpy(outputmeshfile, strs1.str().c_str());

		std::ostringstream strs2;
		strs2 << outputpath << inputfn << "_gpu_new_" << behavior.radius_to_edge_ratio << ".txt";
		outputrepfile = new char[strs2.str().length() + 1];
		strcpy(outputrepfile, strs2.str().c_str());

		refineInputFileOnGPU(
			inputfile,
			&behavior,
			outputmesh ? outputmeshfile : NULL,
			outputrep ? outputrepfile : NULL
		);
	}

	/*internalmesh* drawmesh = behavior.drawmesh = NULL;

	if (drawmesh != NULL && !drawmesh->animation &&
		(drawmesh->iter_seg != -1 || drawmesh->iter_subface != -1 || drawmesh->iter_tet != -1))
	{
		drawMesh(argc, argv, drawmesh);
	}*/

	return 0;
}