#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include "tetgen.h"
#include "CudaRefine.h"
#include "MeshChecker.h"

void scaleCoordinate(
	int numofpoint,
	double* pointlist,
	double maxval
)
{
	double shift[3], scale;
	double min[3], max[3], range[3];
	int i, j;

	for (i = 0; i < 3; i++)
	{
		min[i] = max[i] = pointlist[i];
	}

	for (i = 1; i < numofpoint; i++)
	{
		for (j = 0; j < 3; j++)
		{
			if (pointlist[3 * i + j] < min[j])
				min[j] = pointlist[3 * i + j];
			if (pointlist[3 * i + j] > max[j])
				max[j] = pointlist[3 * i + j];
		}
	}

	double maxr, minc;
	
	for (i = 0; i < 3; i++)
	{
		range[i] = max[i] - min[i];
		if (i == 0)
			maxr = range[i];
		else if (range[i] > maxr)
			maxr = range[i];
	}
	scale = maxval / maxr;

	for (i = 0; i < 3; i++)
	{
		min[i] *= scale;
		shift[i] = 1 - min[i];
	}

	double oldval;
	for (i = 0; i < numofpoint; i++)
	{
		for (j = 0; j < 3; j++)
		{
			oldval = pointlist[3 * i + j];
			pointlist[3 * i + j] = scale*oldval + shift[j];
		}
	}
}

void sortVList(int length, int * indices)
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

void removeDuplicatedTrifaces(
	int numoftriface,
	int* trifacelist,
	int& numofnewface,
	int*& newfacelist
)
{
	bool* dm = new bool[numoftriface];
	int i, j, k, counter = 0;
	for (i = 0; i < numoftriface; i++)
		dm[i] = false;

	int a[3], b[3];
	for (i = 0; i < numoftriface; i++)
	{
		if (dm[i] == true)
			continue;

		for (j = 0; j < 3; j++)
			a[j] = trifacelist[3 * i + j];
		sortVList(3, a);

		for (k = i + 1; k < numoftriface; k++)
		{
			if (dm[k] == true)
				continue;

			for (j = 0; j < 3; j++)
				b[j] = trifacelist[3 * k + j];
			sortVList(3, b);

			if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
				dm[k] = true;
		}
	}

	for (i = 0; i < numoftriface; i++)
	{
		if (!dm[i])
			counter++;
	}

	newfacelist = new int[3 * counter];
	numofnewface = counter;
	counter = 0;
	for (i = 0; i < numoftriface; i++)
	{
		if (!dm[i])
		{
			for (j = 0; j < 3; j++)
				newfacelist[3 * counter + j] = trifacelist[3 * i + j];
			counter++;
		}
	}
}

void saveTr2Mesh(
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist,
	int numoftet,
	int* tetlist,
	char* filename
)
{
	// Save information into the mesh format
	FILE * fp;
	fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("Cannot write to file!");
		exit(0);
	}

	fprintf(fp, "MeshVersionFormatted 1\n");
	fprintf(fp, "Dimension 3\n");

	fprintf(fp, "Vertices\n");
	fprintf(fp, "%d\n", numofpoint);
	for (int i = 0; i < numofpoint; i++)
		fprintf(fp, "%lf %lf %lf 1\n", pointlist[3 * i + 0], pointlist[3 * i + 1], pointlist[3 * i + 2]);

	fprintf(fp, "Triangles\n");
	fprintf(fp, "%d\n", numoftriface);
	for (int i = 0; i < numoftriface; i++)
		fprintf(fp, "%d %d %d 1\n", trifacelist[3 * i + 0] + 1, trifacelist[3 * i + 1] + 1, trifacelist[3 * i + 2] + 1);

	fprintf(fp, "Tetrahedra\n");
	fprintf(fp, "%d\n", numoftet);
	for (int i = 0; i < numoftet; i++)
		fprintf(fp, "%d %d %d %d 3\n", tetlist[4 * i + 0] + 1, tetlist[4 * i + 1] + 1, tetlist[4 * i + 2] + 1, tetlist[4 * i + 3] + 1);
	fprintf(fp, "End\n");

	fclose(fp);
}

void readMesh(
	char* filename,
	int& numofpoint,
	double*& pointlist,
	int& numoftriface,
	int*& trifacelist,
	int& numoftet,
	int*& tetlist
)
{
	FILE *fp;
	fp = fopen(filename, "r");

	if (fp == NULL)
	{
		printf("Cannot find %s!", filename);
		exit(0);
	}

	int ln = 0, ret, index, i, tmp;
	char buf[100];
	double c[3];
	int v[4];
	while (fgets(buf, 100, fp) != NULL) {
		ln++;
		if (ln <= 3)
			continue;
		else if (ln == 4)
		{
			ret = sscanf(buf, "%d", &numofpoint);
			if (!ret)
				break;
			pointlist = new double[3 * numofpoint];
		}
		else if (ln <= 4 + numofpoint)
		{
			ret = sscanf(buf, "%lf %lf %lf %d", c, c + 1, c + 2, &tmp);
			if (!ret)
				break;
			index = ln - 4 - 1;
			for (i = 0; i < 3; i++)
				pointlist[3 * index + i] = c[i];
		}
		else if (ln == 5 + numofpoint)
			continue;
		else if (ln == 6 + numofpoint)
		{
			ret = sscanf(buf, "%d", &numoftriface);
			if (!ret)
				break;
			trifacelist = new int[3 * numoftriface];
		}
		else if (ln <= 6 + numofpoint + numoftriface)
		{
			ret = sscanf(buf, "%d %d %d %d", v, v + 1, v + 2, &tmp);
			if (!ret)
				break;
			index = ln - 6 - numofpoint - 1;
			for (i = 0; i < 3; i++)
				trifacelist[3 * index + i] = v[i] - 1;
		}
		else if (ln == 7 + numofpoint + numoftriface)
			continue;
		else if (ln == 8 + numofpoint + numoftriface)
		{
			ret = sscanf(buf, "%d", &numoftet);
			if (!ret)
				break;
			tetlist = new int[4 * numoftet];
		}
		else if (ln <= 8 + numofpoint + numoftriface + numoftet)
		{
			ret = sscanf(buf, "%d %d %d %d %d", v, v + 1, v + 2, v + 3, &tmp);
			if (!ret)
				break;
			index = ln - 8 - numofpoint - numoftriface - 1;
			for (i = 0; i < 4; i++)
				tetlist[4 * index + i] = v[i] - 1;
		}
	}

	fclose(fp);

	if (!ret)
	{
		printf("Invalid format in %d!\n", filename);
		exit(0);
	}
}

void cutMesh(
	char* infile,
	char* outfile,
	double* plane
)
{
	int numofpoint, numoftriface, numoftet;
	double* pointlist;
	int *trifacelist, *tetlist;
	readMesh(
		infile,
		numofpoint, pointlist,
		numoftriface, trifacelist,
		numoftet, tetlist
	);

	int *newtrifacelist, *newtetlist;
	int numoftriface_new, numoftet_new, counter;
	int i, j, index;
	bool marker;
	double c[3], vec[3];
	double origin[3] = { plane[0], plane[1], plane[2] };
	double norm[3] = { plane[3], plane[4], plane[5] };

	counter = numoftriface;
	for (i = 0; i < numoftriface; i++)
	{
		for (j = 0; j < 3; j++)
		{
			index = trifacelist[3 * i + j];
			c[0] = pointlist[3 * index];
			c[1] = pointlist[3 * index + 1];
			c[2] = pointlist[3 * index + 2];
			vec[0] = c[0] - origin[0];
			vec[1] = c[1] - origin[1];
			vec[2] = c[2] - origin[2];
			if (vec[0] * norm[0] + vec[1] * norm[1] + vec[2] * norm[2] > 0)
			{
				trifacelist[3 * i + 0] = -1;
				trifacelist[3 * i + 1] = -1;
				trifacelist[3 * i + 2] = -1;
				counter--;
				break;
			}
		}
	}
	numoftriface_new = counter;
	counter = 0;
	newtrifacelist = new int[3 * numoftriface_new];
	for (i = 0; i < numoftriface; i++)
	{
		if (trifacelist[3 * i] != -1)
		{
			for (j = 0; j < 3; j++)
			{
				newtrifacelist[3 * counter + j] = trifacelist[3 * i + j];
			}
			counter++;
		}
	}

	counter = numoftet;
	for (i = 0; i < numoftet; i++)
	{
		marker = true;
		for (j = 0; j < 4; j++)
		{
			index = tetlist[4 * i + j];
			c[0] = pointlist[3 * index];
			c[1] = pointlist[3 * index + 1];
			c[2] = pointlist[3 * index + 2];
			vec[0] = c[0] - origin[0];
			vec[1] = c[1] - origin[1];
			vec[2] = c[2] - origin[2];
			if (vec[0] * norm[0] + vec[1] * norm[1] + vec[2] * norm[2] < 0)
			{
				marker = false;
				break;
			}
		}
		if (marker)
		{
			for (j = 0; j < 4; j++)
				tetlist[4 * i + j] = -1;
			counter--;
		}
	}
	numoftet_new = counter;
	counter = 0;
	newtetlist = new int[4 * numoftet_new];
	for (i = 0; i < numoftet; i++)
	{
		if (tetlist[4 * i] != -1)
		{
			for (j = 0; j < 4; j++)
			{
				newtetlist[4 * counter + j] = tetlist[4 * i + j];
			}
			counter++;
		}
	}
	
	saveTr2Mesh(
		numofpoint, pointlist,
		0, NULL,
		0, NULL,
		outfile
	);
}


void saveTr2PLC(
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist,
	int numoftet,
	int* tetlist,
	char* filename1,
	char* filename2
)
{
	// Save information into the PLC format
	FILE * fp;

	// .node file
	fp = fopen(filename1, "w");

	fprintf(fp, "%d 3 0 0\n", numofpoint);
	for (int i = 0; i < numofpoint; i++)
		fprintf(fp, "%d %lf %lf %lf\n", i, pointlist[3 * i + 0], pointlist[3 * i + 1], pointlist[3 * i + 2]);

	fclose(fp);

	// .poly file
	fp = fopen(filename2, "w");

	fprintf(fp, "0 3 0 0\n");
	fprintf(fp, "%d 0\n", numoftriface);
	for (int i = 0; i < numoftriface; i++)
	{
		fprintf(fp, "1  0  0  # %d\n", i);
		fprintf(fp, "3    %d  %d  %d  \n", trifacelist[3 * i], trifacelist[3 * i + 1], trifacelist[3 * i + 2]);
	}
	fprintf(fp, "0\n");
	fprintf(fp, "0\n");

	fclose(fp);
}

void convertPLY2Tr(
	char* filename,
	int header_length,
	int numofpoint,
	double* pointlist,
	int numoftriface,
	int* trifacelist
)
{
	FILE *fp;
	fp = fopen(filename, "r");

	if (fp == NULL)
	{
		printf("Cannot find %s!", filename);
		exit(0);
	}

	int ln = 0, ret, index, i;
	char buf[100];
	double c[3];
	int v[4];
	while (fgets(buf, 100, fp) != NULL) {
		ln++;
		if (ln <= header_length)
			continue;
		else if (ln <= header_length + numofpoint)
		{
			ret = sscanf(buf, "%lf %lf %lf", c, c + 1, c + 2);
			if (!ret)
				break;
			index = ln - header_length - 1;
			for (i = 0; i < 3; i++)
				pointlist[3 * index + i] = c[i];
		}
		else if (ln <= header_length + numofpoint + numoftriface)
		{
			ret = sscanf(buf, "%d %d %d %d", v, v + 1, v + 2, v + 3);
			if (!ret)
				break;
			if (v[0] != 3)
			{
				ret = 0;
				break;
			}
			index = ln - header_length - numofpoint - 1;
			for (i = 0; i < 3; i++)
				trifacelist[3 * index + i] = v[1 + i];
		}
	}

	fclose(fp);

	if (!ret)
	{
		printf("Invalid format in %d!\n", filename);
		exit(0);
	}
}

void convertOFF2Tr(
	char* filename,
	int& numofpoint,
	double*& pointlist,
	int& numoftriface,
	int*& trifacelist
)
{
	FILE *fp;
	fp = fopen(filename, "r");

	if (fp == NULL)
	{
		printf("Cannot find %s!", filename);
		exit(0);
	}

	int ln = 0, ret, index, i;
	char buf[100];
	double c[3];
	int v[4];
	int numoftet;
	while (fgets(buf, 100, fp) != NULL) {
		ln++;
		if (ln == 1)
			continue;
		else if (ln == 2)
		{
			ret = sscanf(buf, "%d %d %d", &numofpoint, &numoftriface, &numoftet);
			if (!ret)
				break;
			pointlist = new double[3 * numofpoint];
			trifacelist = new int[3 * numoftriface];
		}
		else if(ln <= 2 + numofpoint)
		{
			ret = sscanf(buf, "%lf %lf %lf", c, c + 1, c + 2);
			if (!ret)
				break;
			index = ln - 3;
			for (i = 0; i < 3; i++)
				pointlist[3 * index + i] = c[i];
		}
		else if(ln <= 2 + numofpoint + numoftriface)
		{
			ret = sscanf(buf, "%d  %d %d %d", v, v + 1, v + 2, v + 3);
			if (!ret)
				break;
			if (v[0] != 3)
			{
				ret = 0;
				break;
			}
			index = ln - 3 - numofpoint;
			for (i = 0; i < 3; i++)
				trifacelist[3 * index + i] = v[1 + i];
		}
	}

	fclose(fp);

	if (!ret)
	{
		printf("Invalid format in %d!\n", filename);
		exit(0);
	}
}

void convertPLY2Mesh(
	char* infile,
	int header_length,
	int numofpoint,
	int numoftriface,
	char* outfile
)
{
	double* pointlist = new double[3 * numofpoint];
	int* trifacelist = new int[3 * numoftriface];

	convertPLY2Tr(
		infile,
		header_length, numofpoint, pointlist,
		numoftriface, trifacelist);

	int numofnewface, *newfacelist;
	removeDuplicatedTrifaces(
		numoftriface, trifacelist, numofnewface, newfacelist
	);

	scaleCoordinate(
		numofpoint, pointlist, 900
	);

	saveTr2Mesh(
		numofpoint, pointlist,
		numofnewface, newfacelist,
		0, NULL,
		outfile
	);
}

void convertOFF2Mesh(
	char* infile,
	char* outfile
)
{
	int numofpoint, numoftriface;
	double* pointlist;
	int* trifacelist;

	convertOFF2Tr(
		infile, numofpoint, pointlist, numoftriface, trifacelist
	);

	saveTr2Mesh(
		numofpoint, pointlist, numoftriface, trifacelist, 0, NULL, outfile
	);
}

void convertPLC2Mesh(
	char* infile,
	char* outfile
)
{
	tetgenio input;
	input.load_poly(infile);

	int numberoftrifaces = input.numberoffacets;
	int* trifacelist = new int[3 * numberoftrifaces];
	tetgenio::facet * f;
	tetgenio::polygon * p;
	for (int i = 0; i < numberoftrifaces; i++)
	{
		f = &input.facetlist[i];
		p = &f->polygonlist[0];
		for (int j = 0; j < 3; j++)
			trifacelist[3 * i + j] = p->vertexlist[j];
	}

	saveTr2Mesh(
		input.numberofpoints,
		input.pointlist,
		numberoftrifaces,
		trifacelist,
		0,
		NULL,
		outfile
	);
}

void fillOFF(
	char* infile,
	char* outfile
)
{
	int numofpoint, numoftriface;
	double* pointlist;
	int* trifacelist;

	convertOFF2Tr(
		infile, numofpoint, pointlist, numoftriface, trifacelist
	);
}

void convertPLY2PLC(
	char* infile,
	int header_length,
	int numofpoint,
	int numoftriface,
	char* outfile1,
	char* outfile2,
	bool duplicated
)
{
	double* pointlist = new double[3 * numofpoint];
	int* trifacelist = new int[3 * numoftriface];

	convertPLY2Tr(
		infile,
		header_length, numofpoint, pointlist,
		numoftriface, trifacelist);

	int numofnewface, *newfacelist;
	if(duplicated)
		removeDuplicatedTrifaces(
			numoftriface, trifacelist, numofnewface, newfacelist
		);
	else
	{
		numofnewface = numoftriface;
		newfacelist = trifacelist;
	}

	//scaleCoordinate(
	//	numofpoint, pointlist, 900
	//);

	saveTr2PLC(
		numofpoint, pointlist,
		numofnewface, newfacelist,
		0, NULL,
		outfile1,
		outfile2
	);
}

void convertOFF2PLC(
	char* infile,
	char* outfile1,
	char* outfile2
)
{
	int numofpoint, numoftriface;
	double* pointlist;
	int* trifacelist;

	convertOFF2Tr(
		infile, numofpoint, pointlist, numoftriface, trifacelist
	);

	int numofnewface, *newfacelist;
	removeDuplicatedTrifaces(
		numoftriface, trifacelist, numofnewface, newfacelist
	);

	saveTr2PLC(
		numofpoint, pointlist,
		numofnewface, newfacelist,
		0, NULL,
		outfile1,
		outfile2
	);
}

void computeCDTofInputFile(
	char* infile,
	char* outfile,
	int format // 0:PLC, 1:OFF
)
{
	tetgenio in, out;
	if (format == 0)
		in.load_poly(infile);
	else if (format == 1)
		in.load_off(infile);
	else
		exit(0);

	tetrahedralize("VpLO0/0", &in, &out);
	saveTr2Mesh(
		out.numberofpoints,
		out.pointlist,
		out.numberoftrifaces,
		out.trifacelist,
		out.numberoftetrahedra,
		out.tetrahedronlist,
		outfile
	);
}

//void generateTrReport(
//	double* pointlist,
//	int numoftet,
//	int* tetlist,
//	double B,
//	double step,
//	char* file
//)
//{
//	int numofdegtet = 0, numofbadtet = 0, numofslot;
//	double maxratio, minratio;
//	numofslot = B / step + 1;
//	int* counter = new int[numofslot];
//
//	int i, j, slotIdx;
//	int ip[4];
//	double* p[4];
//	double ratio;
//
//	for (i = 0; i < numofslot; i++)
//	{
//		counter[i] = 0;
//	}
//
//	for (i = 0; i < numoftet; i++)
//	{
//		for (j = 0; j < 4; j++)
//		{
//			ip[j] = tetlist[4 * i + j];
//			p[j] = pointlist + 3 * ip[j];
//		}
//		ratio = calRatio(p[0], p[1], p[2], p[3]);
//		if (ratio < 0)
//			numofdegtet++;
//		else
//		{
//			if (i == 0)
//				maxratio = minratio = ratio;
//			else
//			{
//				if (ratio > maxratio)
//					maxratio = ratio;
//				if (ratio < minratio)
//					minratio = ratio;
//			}
//
//			if (ratio > B)
//				numofbadtet++;
//			else
//			{
//				slotIdx = ratio / step;
//				counter[slotIdx]++;
//			}
//		}
//	}
//
//	// Save report
//	//FILE * fp;
//	//fp = fopen(file, "w");
//	//if (fp == NULL)
//	//{
//	//	printf("Cannot write to file!");
//	//	exit(0);
//	//}
//
//	//fprintf(fp, "MeshVersionFormatted 1\n");
//	//fprintf(fp, "Dimension 3\n");
//
//	//fprintf(fp, "Vertices\n");
//	//fprintf(fp, "%d\n", numofpoint);
//	//for (int i = 0; i < numofpoint; i++)
//	//	fprintf(fp, "%lf %lf %lf 1\n", pointlist[3 * i + 0], pointlist[3 * i + 1], pointlist[3 * i + 2]);
//}

void generateTrReport(
	int numofpoint,
	double* pointlist,
	int numofsubseg,
	int numofsubface,
	int numoftet,
	int* tetlist,
	double B,
	double step,
	char* file,
	double* times
)
{
	int numofbadtet, numofslot;
	double maxangle, minangle;

	numofbadtet = countBadTets(pointlist, tetlist, numoftet, B);

	numofslot = 180 / step + 1;
	int* counter = new int[numofslot];

	int i, j, slotIdx;
	int ip[4];
	double* p[4];
	double alldihed[6];

	for (i = 0; i < numofslot; i++)
	{
		counter[i] = 0;
	}

	for (i = 0; i < numoftet; i++)
	{
		for (j = 0; j < 4; j++)
		{
			ip[j] = tetlist[4 * i + j];
			p[j] = pointlist + 3 * ip[j];
		}
		
		calDihedral(p, alldihed);
		for (j = 0; j < 6; j++)
		{
			slotIdx = alldihed[j] / step;
			counter[slotIdx]++;
		}
	}

	// Save report
	if (file != NULL)
	{
		FILE * fp;
		fp = fopen(file, "w");
		if (fp != NULL)
		{
			fprintf(fp, "Number of points = %d\n", numofpoint);
			fprintf(fp, "Number of subfaces = %d\n", numofsubface);
			fprintf(fp, "Number of segments = %d\n", numofsubseg);
			fprintf(fp, "Number of tetrahedra = %d\n", numoftet);
			fprintf(fp, "Number of bad tetrahedra = %d\n", numofbadtet);
			fprintf(fp, "Total time = %lf\n", times[0]);
			fprintf(fp, "CPU time = %lf\n", times[1]);
			fprintf(fp, "  Splitting segments time = %lf\n", times[2]);
			fprintf(fp, "  Splitting subfaces time = %lf\n", times[3]);
			fprintf(fp, "  Splitting bad tets time = %lf\n", times[4]);
			fprintf(fp, "GPU time = %lf\n", times[5]);
			fprintf(fp, "  Reconstruction time = %lf\n", times[6]);
			fprintf(fp, "  Initialization time = %lf\n", times[7]);
			fprintf(fp, "  Splitting segments time = %lf\n", times[8]);
			fprintf(fp, "  Splitting subfaces time = %lf\n", times[9]);
			fprintf(fp, "  Splitting bad tets time = %lf\n", times[10]);
			fprintf(fp, "  Output final mesh  time = %lf\n", times[11]);

			for (i = 0; i < numofslot; i++)
			{
				fprintf(fp, "%lf %d\n", i*step, counter[i]);
			}
		}
		fclose(fp);
	}

	printf("Number of points = %d\n", numofpoint);
	printf("Number of subfaces = %d\n", numofsubface);
	printf("Number of segments = %d\n", numofsubseg);
	printf("Number of tetrahedra = %d\n", numoftet);
	printf("Number of bad tetrahedra = %d\n", numofbadtet);
	printf("Total time = %lf\n", times[0]);
	printf("CPU time = %lf\n", times[1]);
	printf("  Splitting segments time = %lf\n", times[2]);
	printf("  Splitting subfaces time = %lf\n", times[3]);
	printf("  Splitting bad tets time = %lf\n", times[4]);
	printf("GPU time = %lf\n", times[5]);
	printf("  Reconstruction time = %lf\n", times[6]);
	printf("  Initialization time = %lf\n", times[7]);
	printf("  Splitting segments time = %lf\n", times[8]);
	printf("  Splitting subfaces time = %lf\n", times[9]);
	printf("  Splitting bad tets time = %lf\n", times[10]);
	printf("  Output final mesh  time = %lf\n", times[11]);
}

void refineInputFileOnCPU(
	char* infile,
	int format,
	double B,
	char* outmesh,
	char* outdata
)
{
	tetgenio input_plc, output_cdt;
	if (format == 0)
		input_plc.load_poly(infile);
	else if (format == 1)
		input_plc.load_off(infile);
	else
		exit(0);

	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
	clock_t tv[2];
	tv[0] = clock();

	// CPU
	double times[12];
	char *com;
	std::ostringstream strs;
	strs << "VpnLq" << B << "O0/0";

	std::string fn = strs.str();
	com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	tetrahedralize(com, &input_plc, &output_cdt);

	tv[1] = clock();
	times[0] = (double)(tv[1] - tv[0]);
	times[1] = times[0];
	times[2] = output_cdt.refine_times[0];
	times[3] = output_cdt.refine_times[1];
	times[4] = output_cdt.refine_times[2];
	for (int i = 1; i <= 7; i++)
		times[4 + i] = 0;

	// Output mesh
	if (outmesh != NULL)
		saveTr2Mesh(
			output_cdt.numberofpoints,
			output_cdt.pointlist,
			output_cdt.numberoftrifaces,
			output_cdt.trifacelist,
			output_cdt.numberoftetrahedra,
			output_cdt.tetrahedronlist,
			outmesh
		);

	// Output statistic
	if (outdata != NULL)
	{
		generateTrReport(
			output_cdt.numberofpoints,
			output_cdt.pointlist,
			output_cdt.numberofedges,
			output_cdt.numberoftrifaces,
			output_cdt.numberoftetrahedra,
			output_cdt.tetrahedronlist,
			B,
			1,
			outdata,
			times
		);
	}
}

void refineInputFileOnGPU(
	char* infile,
	MESHBH* input_behavior,
	char* outmesh,
	char* outdata
)
{
	// Set input behavior
	if (input_behavior->R2)
		input_behavior->filtermode = 2;
	else
		input_behavior->filtermode = 1;

	if (input_behavior->R4)
	{
		input_behavior->vecmode = 2;
	}
	else
	{
		input_behavior->vecmode = 1;
	}

	if (input_behavior->R5)
	{
		input_behavior->cpumode = 2;
		//input_behavior->cavitymode = 2;
	}
	else
	{
		input_behavior->cpumode = 1;
		//input_behavior->cavitymode = 1;
	}

	// Read Input
	tetgenio input_plc, input_cdt;
	if (input_behavior->fileformat == 0)
		input_plc.load_poly(infile);
	else if (input_behavior->fileformat == 1)
		input_plc.load_off(infile);
	else
		exit(0);

	// CPU
	clock_t tv[3];
	double times[12];
	tv[0] = clock();
	char *com;
	std::ostringstream strs;
	if (input_behavior->cpumode == 1)
		strs << "VQGpnLO0/0";
	else if (input_behavior->cpumode == 2)
		strs << "VQGpnLq" << input_behavior->radius_to_edge_ratio << "D2O0/0";
	else
		exit(0);

	std::string fn = strs.str();
	com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	tetrahedralize(com, &input_plc, &input_cdt);
	tv[1] = clock();
	times[1] = (double)(tv[1] - tv[0]);
	if(input_behavior->cpumode == 1)
		times[2] = times[3] = times[4] = 0;
	else
	{
		times[2] = input_cdt.refine_times[0];
		times[3] = input_cdt.refine_times[1];
		times[4] = input_cdt.refine_times[2];
	}

	//GPU
	tv[1] = clock();
	int out_numofpoint;
	REAL* out_pointlist;
	int out_numofedge;
	int* out_edgelist;
	int out_numoftriface;
	int* out_trifacelist;
	int out_numoftet;
	int* out_tetlist;

	MESHIO  input_gpu;
	input_gpu.numofpoints = input_cdt.numberofpoints;
	input_gpu.pointlist = input_cdt.pointlist;
	input_gpu.numofedges = input_cdt.numberofedges;
	input_gpu.edgelist = input_cdt.edgelist;
	input_gpu.numoftrifaces = input_cdt.numberoftrifaces;
	input_gpu.trifacelist = input_cdt.trifacelist;
	input_gpu.numoftets = input_cdt.numberoftetrahedra;
	input_gpu.tetlist = input_cdt.tetrahedronlist;
	input_gpu.interneigh = input_cdt.interneigh;
	if (input_gpu.interneigh)
	{
		input_gpu.interpointradius = input_cdt.interpointradius;
		input_gpu.intertet2tetlist = input_cdt.intertet2tetlist;
	}

	GPU_Refine_3D(
		&input_gpu,
		input_behavior,
		out_numofpoint,
		out_pointlist,
		out_numofedge,
		out_edgelist,
		out_numoftriface,
		out_trifacelist,
		out_numoftet,
		out_tetlist
	);

	tv[2] = clock();
	times[0] = (double)(tv[2] - tv[0]);
	times[5] = (double)(tv[2] - tv[1]);

	for (int i = 1; i <= 6; i++)
	{
		times[5 + i] = input_behavior->times[i - 1];
	}

	// Output mesh
	if (outmesh != NULL)
		saveTr2Mesh(
			out_numofpoint,
			out_pointlist,
			out_numoftriface,
			out_trifacelist,
			out_numoftet,
			out_tetlist,
			outmesh
		);

	// Output statistic
	generateTrReport(
		out_numofpoint,
		out_pointlist,
		out_numofedge,
		out_numoftriface,
		out_numoftet,
		out_tetlist,
		input_behavior->radius_to_edge_ratio,
		1,
		outdata,
		times
	);
}