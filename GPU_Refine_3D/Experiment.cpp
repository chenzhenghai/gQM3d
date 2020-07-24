#include <time.h>
#include <helper_timer.h>
#include "Experiment.h"
#include "InputGenerator.h"
#include "MeshChecker.h"

void generatePLCs()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio out;
	printf("Generating PLC files......\n");
	for (dist = (Distribution)3; dist <= (Distribution)3; dist = (Distribution)((int)dist + 1))
	{
		//for (numofpoint = 10000; numofpoint <= 50000; numofpoint += 10000)
		numofpoint = 100;
		{
			//for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.2; numoftri += numofpoint*0.05)
			{
				numoftri = 10;
				printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f\n",
					numofpoint, numoftri, numofedge, seed, dist, minarea);

				if (readInputPLCFile(numofpoint, numoftri, numofedge, seed, dist, minarea, &out))
				{
					printf("Already have!\n");
					//out.deinitialize();
				}
				else
				{
					generateInputPLCFile(
						numofpoint,
						numoftri,
						numofedge,
						seed, dist, minarea);
				}
				printf("\n");
			}
		}
	}
}

void generateCDTs()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio out;
	printf("Generating CDT files......\n");
	for (dist = (Distribution)0; dist <= (Distribution)5; dist = (Distribution)((int)dist + 1))
	{
		//for (numofpoint = 10000; numofpoint <= 50000; numofpoint += 10000)
		numofpoint = 10000;
		{
			for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.2; numoftri += numofpoint*0.05)
			{
				printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f\n",
					numofpoint, numoftri, numofedge, seed, dist, minarea);
				if (readInputCDTFile(numofpoint, numoftri, numofedge, seed, dist, minarea, &out))
				{
					printf("Already have!\n");
					//out.deinitialize();
				}
				else
				{
					generateInputCDTFile(
						numofpoint,
						numoftri,
						numofedge,
						seed, dist, minarea);
				}
				printf("\n");
			}
		}
	}
}

bool readResultFile(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	double radio
)
{
	// Prepare for filename
	std::ostringstream strs;
	strs << "result/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
		<< "_q" << radio << "_noflip" << ".txt";
	std::string fn = strs.str();
	char *fileName = new char[fn.length() + 1];
	strcpy(fileName, fn.c_str());

	// Try to open

	FILE *fp;
	fp = fopen(fileName, "r");
	if (fp == NULL)
		return false;
	fclose(fp);
	return true;
}

void refineMeshCPU()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio in, out;
	printf("Refining Mesh......\n");
	//for (dist = (Distribution)0; dist <= (Distribution)5; dist = (Distribution)((int)dist + 1))
	{
		dist = (Distribution)0;
		//for (numofpoint = 10000; numofpoint <= 50000; numofpoint += 10000)
		numofpoint = 10000;
		{
			for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.2; numoftri += numofpoint*0.05)
			{
				// Set up parameter
				dist = (Distribution)0;
				double radius_to_edge_ratio = 1.414;

				printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f, radius-to-edge-radio = %lf\n",
					numofpoint, numoftri, numofedge, seed, dist, minarea, radius_to_edge_ratio);

				// Run refinement
				if (readResultFile(numofpoint,numoftri,numofedge,seed,dist,minarea,radius_to_edge_ratio)) // already have result file
				{
					printf("Found result file!\n");
				}
				else if (readInputPLCFile(numofpoint, numoftri, numofedge, seed, dist, minarea, &in))
				{
					printf("Refinement in process......\n");

					// Prepare common line
					char *com;
					{
						std::ostringstream strs;
						strs << "VpLq" << radius_to_edge_ratio << "O0/0";
						std::string fn = strs.str();
						com = new char[fn.length() + 1];
						strcpy(com, fn.c_str());
					}

					// Calculate quality mesh
					StopWatchInterface *timer = 0; // timer
					sdkCreateTimer(&timer);
					double cpu_time;
					sdkResetTimer(&timer);
					sdkStartTimer(&timer);
					tetrahedralize(com, &in, &out); // Supposed to terminate
					sdkStopTimer(&timer);
					cpu_time = sdkGetTimerValue(&timer);

					// Prepare for fileName
					char *fileName;
					{
						std::ostringstream strs;
						strs << "result/d" << dist << "_s" << seed << "_a" << minarea
							<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
							<< "_q" << radius_to_edge_ratio << "_noflip" << ".txt";
						std::string fntmp = strs.str();
						fileName = new char[fntmp.length() + 1];
						strcpy(fileName, fntmp.c_str());
					}

					// Save information into files
					FILE * fp;
					fp = fopen(fileName, "w");
					fprintf(fp, "Number of points = %d\n", out.numberofpoints);
					fprintf(fp, "Number of subfaces = %d\n", out.numberoftrifaces);
					fprintf(fp, "Number of segments = %d\n", out.numberofedges);
					fprintf(fp, "Number of tetrahedra = %d\n", out.numberoftetrahedra);
					fprintf(fp, "Total time = %lf\n", cpu_time);

					fclose(fp);

					// Release memeory
					//in.deinitialize();
					//out.deinitialize();
				}
				else
				{
					printf("Failed to read PLC file, skip!\n");
				}
				printf("\n");
			}
		}
	}
}

void generatePLCs_without_acute_angles()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio out;
	printf("Generating PLC files......\n");
	for (dist = (Distribution)0; dist <= (Distribution)3; dist = (Distribution)((int)dist + 1))
	{
		//dist = (Distribution)1;
		//for (numofpoint = 15000; numofpoint <= 30000; numofpoint += 1000)
		numofpoint = 100;
		{
			//for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.25; numoftri += numofpoint*0.05)
			{
				numoftri = 25;
				printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f\n",
					numofpoint, numoftri, numofedge, seed, dist, minarea);

				if (readInputPLCFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &out))
				{
					printf("Already have!\n");
					//out.deinitialize();
				}
				else
				{
					generateInputPLCFile_without_acute_angles(
						numofpoint,
						numoftri,
						numofedge,
						seed, dist, minarea);
				}
				printf("\n");
			}
		}
	}
}

void generateCDTs_without_acute_angles()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio out;
	printf("Generating CDT files......\n");
	//for (dist = (Distribution)0; dist <= (Distribution)5; dist = (Distribution)((int)dist + 1))
	{
		dist = (Distribution)0;
		//for (numofpoint = 10000; numofpoint <= 30000; numofpoint += 5000)
		{
			numofpoint = 100;
			for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.25; numoftri += numofpoint*0.05)
			{
				printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f\n",
					numofpoint, numoftri, numofedge, seed, dist, minarea);
				if (readInputCDTFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &out))
				{
					printf("Already have!\n");
					//out.deinitialize();
				}
				else
				{
					generateInputCDTFile_without_acute_angles(
						numofpoint,
						numoftri,
						numofedge,
						seed, dist, minarea);
				}
				printf("\n");
			}
		}
	}
}

bool readResultFile_without_acute_angles(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	double radio
)
{
	// Prepare for filename
	std::ostringstream strs;
	strs << "result_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
		<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
		<< "_q" << radio << "_noflip" << ".txt";
	std::string fn = strs.str();
	char *fileName = new char[fn.length() + 1];
	strcpy(fileName, fn.c_str());

	// Try to open

	FILE *fp;
	fp = fopen(fileName, "r");
	if (fp == NULL)
		return false;
	fclose(fp);
	return true;
}

bool readResultFile_without_acute_angles_tetgen(
	int numofpoint,
	int numoftri,
	int numofedge,
	int seed,
	Distribution dist,
	double minarea,
	double radio,
	double * data
)
{

}

void refineMeshCPU_without_acute_angles()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 1;
	Distribution dist;
	double minarea = 0;
	tetgenio in, out;
	double radius_to_edge_radio = 2;
	printf("Refining PLC......\n");
	//for (radius_to_edge_radio = 1.6; radius_to_edge_radio <= 2.0; radius_to_edge_radio += 0.2)
	{
		//for (dist = (Distribution)0; dist <= (Distribution)3; dist = (Distribution)((int)dist + 1))
		{
			dist = (Distribution)1;
			//for (numofpoint = 15000; numofpoint <= 15000; numofpoint += 5000)
			{
				numofpoint = 35000;
				for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.25; numoftri += numofpoint*0.05)
				{
					//numoftri = 3750;
					// Set up parameter

					printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f, radius-to-edge-radio = %lf\n",
						numofpoint, numoftri, numofedge, seed, dist, minarea, radius_to_edge_radio);

					// Run refinement
					if (readResultFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, radius_to_edge_radio)) // already have result file
					{
						printf("Found result file!\n");
					}
					else if (readInputPLCFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &in))
					{
						printf("Refinement in process......\n");
						time_t rawtime;
						struct tm * timeinfo;
						time(&rawtime);
						timeinfo = localtime(&rawtime);
						printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

						// Prepare common line
						char *com;
						{
							std::ostringstream strs;
							strs << "pLq" << radius_to_edge_radio << "O0/0";
							std::string fn = strs.str();
							com = new char[fn.length() + 1];
							strcpy(com, fn.c_str());
						}

						// Calculate quality mesh
						StopWatchInterface *timer = 0; // timer
						sdkCreateTimer(&timer);
						double cpu_time;
						sdkResetTimer(&timer);
						sdkStartTimer(&timer);
						tetrahedralize(com, &in, &out); // Supposed to terminate
						sdkStopTimer(&timer);
						cpu_time = sdkGetTimerValue(&timer);

						int numofbadtets = countBadTets(out.pointlist, out.tetrahedronlist, out.numberoftetrahedra, radius_to_edge_radio);

						// Prepare for fileName
						char *fileName;
						{
							std::ostringstream strs;
							strs << "result_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
								<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
								<< "_q" << radius_to_edge_radio << "_noflip" << ".txt";
							std::string fntmp = strs.str();
							fileName = new char[fntmp.length() + 1];
							strcpy(fileName, fntmp.c_str());
						}

						// Save information into files
						FILE * fp;
						fp = fopen(fileName, "w");
						fprintf(fp, "Number of points = %d\n", out.numberofpoints);
						fprintf(fp, "Number of subfaces = %d\n", out.numberoftrifaces);
						fprintf(fp, "Number of segments = %d\n", out.numberofedges);
						fprintf(fp, "Number of tetrahedra = %d\n", out.numberoftetrahedra);
						fprintf(fp, "Number of bad tetrahedra = %d\n", numofbadtets);
						fprintf(fp, "Total time = %lf\n", cpu_time);
						fprintf(fp, "  Splitting segments time = %lf\n", out.refine_times[0]);
						fprintf(fp, "  Splitting subfaces time = %lf\n", out.refine_times[1]);
						fprintf(fp, "  Splitting bad tets time = %lf\n", out.refine_times[2]);
						fclose(fp);

						// Release memeory
						//in.deinitialize();
						//out.deinitialize();
					}
					else
					{
						printf("Failed to read PLC file, skip!\n");
					}
					printf("\n");
				}
			}
		}
	}
}

void saveGPUResult_without_acute_angles(
	int numofpoint, int numoftri, int numofedge,
	int seed, Distribution dist, double minarea, double radio, 
	int outnumofpoint, int outnumoftrifaces, int outnumofedges, 
	int outnumoftets, int outnumofbadtets, double* times, int mode
)
{
	// Prepare for fileName
	char *fileName;
	{
		std::ostringstream strs;
		if(mode == 1)
			strs << "result_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
				<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
				<< "_q" << radio << "_noflip_gpu";
		else if(mode == 2)
			strs << "result_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
			<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
			<< "_q" << radio << "_noflip_gpu_star";
		strs << ".txt";
		std::string fntmp = strs.str();
		fileName = new char[fntmp.length() + 1];
		strcpy(fileName, fntmp.c_str());
	}

	// Save information into files
	FILE * fp;
	fp = fopen(fileName, "w");
	fprintf(fp, "Number of points = %d\n", outnumofpoint);
	fprintf(fp, "Number of subfaces = %d\n", outnumoftrifaces);
	fprintf(fp, "Number of segments = %d\n", outnumofedges);
	fprintf(fp, "Number of tetrahedra = %d\n", outnumoftets);
	fprintf(fp, "Number of bad tetrahedra = %d\n", outnumofbadtets);
	fprintf(fp, "Total time = %lf\n", times[11]);
	fprintf(fp, "CPU time = %lf\n", times[9]);
	fprintf(fp, "  Splitting segments time = %lf\n", times[6]);
	fprintf(fp, "  Splitting subfaces time = %lf\n", times[7]);
	fprintf(fp, "  Splitting bad tets time = %lf\n", times[8]);
	fprintf(fp, "GPU time = %lf\n", times[10]);
	fprintf(fp, "  Reconstruction time = %lf\n", times[0]);
	fprintf(fp, "  Initialization time = %lf\n", times[1]);
	fprintf(fp, "  Splitting segments time = %lf\n", times[2]);
	fprintf(fp, "  Splitting subfaces time = %lf\n", times[3]);
	fprintf(fp, "  Splitting bad tets time = %lf\n", times[4]);
	fprintf(fp, "  Output final mesh  time = %lf\n", times[5]);
	fclose(fp);
}

void experiment_statistic()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	double radius_to_edge_radio = 1.414;

	for (dist = (Distribution)0; dist <= (Distribution)3; dist = (Distribution)((int)dist + 1))
	{
		for (numofpoint = 15000; numofpoint <= 30000; numofpoint += 5000)
		{
			for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.25; numoftri += numofpoint*0.05)
			{
				
			}
		}
	}
}