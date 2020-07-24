//#define CGAL_LINKED_WITH_TBB
//#define CGAL_CONCURRENT_MESH_3
//#define NOMINMAX

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Delaunay_Triangulation_3.h>
#include <CGAL/Timer.h>
#include "InputGenerator.h"
#include "MeshChecker.h"

// Domain 
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<
	Tr, Mesh_domain::Corner_index, Mesh_domain::Curve_index> C3t3;
typedef C3t3::Triangulation Tr3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;

void debug_CGAL_output(C3t3& c3t3, double ratio);
double debug_distance(double* p1, double* p2)
{
	double dis = (p1[0] - p2[0])*(p1[0] - p2[0]) +
		(p1[1] - p2[1])*(p1[1] - p2[1]) + (p1[2] - p2[2])*(p1[2] - p2[2]);
	dis = sqrt(dis);
	return dis;
}

double debug_shortestedge(double* p1, double* p2, double* p3, double* p4)
{
	double *p[4], len, dis;
	p[0] = p1; p[1] = p2; p[2] = p3; p[3] = p4;
	for (int i = 0; i < 3; i++)
	{
		for (int j = i + 1; j < 4; j++)
		{
			dis = debug_distance(p[i], p[j]);
			if (i == 0 && j == 1)
				len = dis;
			else if (dis < len)
				len = dis;
		}
	}

	return len;
}

int cgalTest()
{
	std::cout.precision(17);
	std::cerr.precision(17);
	const char* fname = "cgal_test/horizons.off";
	std::ifstream input(fname);
	const char* fname2 = "cgal_test/horizons-domain.off";
	std::ifstream input2(fname2);
	Polyhedron sm, smbounding;
	input >> sm;
	input2 >> smbounding;
	if (input.fail()) {
		std::cerr << "Error: Cannot read file " << fname << std::endl;
		return EXIT_FAILURE;
	}
	CGAL::Timer t;
	t.start();
	// Create domain
	Mesh_domain domain(sm, smbounding);
	
	// Get sharp features
	domain.detect_features();
	
	// Mesh criteria
	Mesh_criteria criteria(edge_size = 0.025,
		facet_angle = 25, facet_size = 0.05, facet_distance = 0.005,
		cell_radius_edge_ratio = 3, cell_size = 0.05);

	// Mesh generation
	C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria,
		no_perturb(), no_exude());
	std::cerr << t.time() << " sec." << std::endl;

	debug_CGAL_output(c3t3, 3.0);

	// Output
	std::ofstream medit_file("out_1.mesh");
	c3t3.output_to_medit(medit_file);
}

int cgalTest2()
{
	std::cout.precision(17);
	std::cerr.precision(17);
	const char* fname = "input_off_without_acute_angle/d0_s0_a0_p10000_t500_e0.off";
	std::ifstream input(fname);
	const char* fname2 = "input_off_without_acute_angle/bounding_box.off";
	std::ifstream input2(fname2);
	Polyhedron sm, smbounding;
	input >> sm;
	input2 >> smbounding;
	if (input.fail()) {
		std::cerr << "Error: Cannot read file " << fname << std::endl;
		return EXIT_FAILURE;
	}
	printf("%d, %d\n", sm.size_of_vertices(), sm.size_of_facets());
	printf("%d, %d\n", smbounding.size_of_vertices(), smbounding.size_of_facets());

	CGAL::Timer t;
	t.start();
	// Create domain
	Mesh_domain domain(sm, smbounding);
	// Get sharp features
	domain.detect_features();
	// Mesh criteria
	Mesh_criteria criteria(
		edge_size = 100,
		facet_angle = 25,
		cell_radius_edge_ratio = 1.414);

	// Mesh generation
	C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria,
		no_perturb(), no_exude());
	std::cerr << t.time() << " sec." << std::endl;

	// Output mesh
	printf("Number of points = %d\n", c3t3.triangulation().number_of_vertices());
	printf("Number of subfaces = %d\n", c3t3.triangulation().number_of_finite_facets());
	printf("Number of segments = %d\n", c3t3.triangulation().number_of_finite_edges());
	printf("Number of tetrahedra = %d\n", c3t3.triangulation().number_of_finite_cells());
	//std::ofstream medit_file("out.mesh");
	//c3t3.output_to_medit(medit_file);
	std::ofstream off_file("out_2.off");
	c3t3.output_boundary_to_off(off_file);
}

int cgalTest3()
{
	std::cout.precision(17);
	std::cerr.precision(17);
	CGAL::Timer t;
	t.start();

	int numofpoint = 10000, numoftri = 2000, numofedge = 0;
	int seed = 0;
	Distribution dist = (Distribution)0;
	double minarea = 0;
	tetgenio out;

	const char* fname = "input_off_without_acute_angle/d0_s0_a0_p10000_t2000_e0.off";
	std::ifstream input(fname);
	const char* fname2 = "input_off_without_acute_angle/bounding_box.off";
	std::ifstream input2(fname2);
	Polyhedron sm, smbounding;
	input >> sm;
	input2 >> smbounding;
	if (input.fail()) {
		std::cerr << "Error: Cannot read file " << fname << std::endl;
		return EXIT_FAILURE;
	}
	printf("%d, %d\n", sm.size_of_vertices(), sm.size_of_facets());
	printf("%d, %d\n", smbounding.size_of_vertices(), smbounding.size_of_facets());

	// Create domain
	Mesh_domain domain(sm, smbounding);

	// Get sharp features
	//domain.detect_features();
	bool ret = readInputPLCFile_without_acute_angles(
		numofpoint, numoftri, numofedge, seed, dist, minarea, &out);
	if (!ret)
		exit(0);

	// Add input points as corners
	std::vector<K::Point_3> points;
	for (int i = 8; i < out.numberofpoints; i++)
	{
		points.push_back(
			K::Point_3(out.pointlist[3 * i], out.pointlist[3 * i + 1], out.pointlist[3 * i + 2]));
	}
	domain.add_corners(points.begin(), points.end());

	tetgenio::facet *f,  *f1;
	tetgenio::polygon *p, *p1;
	std::vector<std::vector<K::Point_3>> segments;
	// Add segments of the bounding box
	f = &out.facetlist[0];
	p = &f->polygonlist[0];
	f1 = &out.facetlist[1];
	p1 = &f1->polygonlist[0];
	for (int i = 0; i < 4; i++)
	{
		int v[4];
		v[0] = p->vertexlist[i];
		v[1] = p->vertexlist[(i + 1) % 4];
		v[2] = p1->vertexlist[i];
		v[3] = p1->vertexlist[(i + 1) % 4];
		std::vector<K::Point_3> segment0, segment1, segment2;
		segment0.push_back(
			K::Point_3(out.pointlist[3 * v[0]], out.pointlist[3 * v[0] + 1], out.pointlist[3 * v[0] + 2]));
		segment0.push_back(
			K::Point_3(out.pointlist[3 * v[1]], out.pointlist[3 * v[1] + 1], out.pointlist[3 * v[1] + 2]));
		segment1.push_back(
			K::Point_3(out.pointlist[3 * v[2]], out.pointlist[3 * v[2] + 1], out.pointlist[3 * v[2] + 2]));
		segment1.push_back(
			K::Point_3(out.pointlist[3 * v[3]], out.pointlist[3 * v[3] + 1], out.pointlist[3 * v[3] + 2]));
		segment2.push_back(
			K::Point_3(out.pointlist[3 * v[0]], out.pointlist[3 * v[0] + 1], out.pointlist[3 * v[0] + 2]));
		segment2.push_back(
			K::Point_3(out.pointlist[3 * v[2]], out.pointlist[3 * v[2] + 1], out.pointlist[3 * v[2] + 2]));
		segments.push_back(segment0);
		segments.push_back(segment1);
		segments.push_back(segment2);
	}

	// Add segments of the inside subfaces
	for (int i = 6; i < out.numberoffacets; i++)
	{
		f = &out.facetlist[i];
		p = &f->polygonlist[0];
		for (int j = 0; j < 4; j++)
		{
			int v[2];
			v[0] = p->vertexlist[j];
			v[1] = p->vertexlist[(j + 1) % 4];
			std::vector<K::Point_3> segment;
			segment.push_back(
				K::Point_3(out.pointlist[3 * v[0]], out.pointlist[3 * v[0] + 1], out.pointlist[3 * v[0] + 2]));
			segment.push_back(
				K::Point_3(out.pointlist[3 * v[1]], out.pointlist[3 * v[1] + 1], out.pointlist[3 * v[1] + 2]));
			segments.push_back(segment);
		}
	}

	domain.add_features(segments.begin(), segments.end());

	// Mesh criteria
	Mesh_criteria criteria(
		edge_size = 5,
		facet_angle = 25, facet_distance = 1,
		cell_radius_edge_ratio = 1.414);

	// Mesh generation
	C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria,
		no_perturb(), no_exude());
	std::cerr << t.time() << " sec." << std::endl;

	// Output mesh
	printf("Number of points = %d\n", c3t3.triangulation().number_of_vertices());
	printf("Number of subfaces = %d\n", c3t3.number_of_facets());
	printf("Number of segments = %d\n", c3t3.number_of_edges());
	printf("Number of tetrahedra = %d\n", c3t3.number_of_cells());
	std::ofstream medit_file("out_3.mesh");
	c3t3.output_to_medit(medit_file);
	//std::ofstream off_file("out_3.off");
	//c3t3.output_boundary_to_off(off_file);
}

void generateOFFs_without_acute_angles()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio out;
	printf("Generating OFF files......\n");
	//for (dist = (Distribution)0; dist <= (Distribution)5; dist = (Distribution)((int)dist + 1))
	{
		dist = (Distribution)3;
		for (numofpoint = 15000; numofpoint <= 15000; numofpoint += 5000)
		{
			for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.2; numoftri += numofpoint*0.05)
			{
				printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f\n",
					numofpoint, numoftri, numofedge, seed, dist, minarea);

				if (readInputPLCFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &out))
				{
					generateInputOFFFile_without_acute_angles2(numofpoint, numoftri, numofedge, seed, dist, minarea, &out);
					//break;
				}
				else
				{
					printf("Failed to read PLC files!\n");
				}
			}
		}
	}
}

bool readResultFile_CGAL_without_acute_angles(
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
		<< "_q" << radio << "_cgal" << ".txt";
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

void debug_CGAL_output(C3t3& c3t3, double ratio)
{
	int number_of_vertices = c3t3.triangulation().number_of_vertices();
	int number_of_facets = c3t3.triangulation().number_of_finite_facets();
	int number_of_edges = c3t3.triangulation().number_of_finite_edges();
	int number_of_cells = c3t3.triangulation().number_of_finite_cells();

	// Output medit
	char *fileName = "out_debug_badtets.mesh";
	FILE * fp;
	fp = fopen(fileName, "w");
	fprintf(fp, "MeshVersionFormatted 1\n");
	fprintf(fp, "Dimension 3\n");

	std::map<Tr3::Vertex_handle, int> vertices;
	double* pointlist = new double[3 * number_of_vertices];
	int* tetlist = new int[4 * number_of_cells];

	int index = 0;
	fprintf(fp, "Vertices\n");
	fprintf(fp, "%d\n", number_of_vertices);
	for (
		Tr3::Finite_vertices_iterator vit = c3t3.triangulation().finite_vertices_begin(),
		end = c3t3.triangulation().finite_vertices_end(); vit != end; ++vit)
	{
		vertices[vit] = index;
		pointlist[3 * index + 0] = vit->point().x();
		pointlist[3 * index + 1] = vit->point().y();
		pointlist[3 * index + 2] = vit->point().z();
		fprintf(fp, "%lf %lf %lf\n", vit->point().x(), vit->point().y(), vit->point().z());
		index++;
	}

	index = 0;
	int badindex = 0;
	double ccent[3], dis;
	fprintf(fp, "Tetrahedra\n");
	printf("\n");
	for (
		Tr3::Finite_cells_iterator cit = c3t3.triangulation().finite_cells_begin(),
		end = c3t3.triangulation().finite_cells_end(); cit != end; ++cit)
	{
		int v[4];
		for (int i = 0; i < 4; i++)
		{
			v[i] = vertices.find(cit->vertex(i))->second;
			tetlist[4 * index + i] = v[i];
		}
		if (isBadTet(
				pointlist + 3 * v[0], pointlist + 3 * v[1], 
				pointlist + 3 * v[2], pointlist + 3 * v[3], ratio, ccent))
		{
			if (badindex < 50)
			{
				printf("Tet %d - Center %lf %lf %lf\n", index, ccent[0], ccent[1], ccent[2]);
				printf("Shortest edge length = %lf\n",
					debug_shortestedge(
						pointlist + 3 * v[0], pointlist + 3 * v[1],
						pointlist + 3 * v[2], pointlist + 3 * v[3]));
				for (int i = 0; i < 4; i++)
				{
					dis = debug_distance(ccent, pointlist + 3 * v[i]);
					printf("Vertex %d - %lf %lf %lf - Distance %lf\n", v[i],
						pointlist[3 * v[i]], pointlist[3 * v[i] + 1], pointlist[3 * v[i] + 2],
						dis);
				}
			}
			badindex++;
		}
		index++;
	}
	fprintf(fp, "%d\n", badindex);
	printf("Number of tets = %d, bad tets = %d, percentage = %lf\n", index, badindex, badindex*1.0 / index);

	for (
		Tr3::Finite_cells_iterator cit = c3t3.triangulation().finite_cells_begin(),
		end = c3t3.triangulation().finite_cells_end(); cit != end; ++cit)
	{
		int v[4];
		for (int i = 0; i < 4; i++)
		{
			v[i] = vertices.find(cit->vertex(i))->second;
		}
		if (isBadTet(
			pointlist + 3 * v[0], pointlist + 3 * v[1],
			pointlist + 3 * v[2], pointlist + 3 * v[3], ratio, ccent))
		{
			fprintf(fp, "%d %d %d %d\n", v[0] + 1, v[1] + 1, v[2] + 1, v[3] + 1);
		}
	}
	fprintf(fp, "End\n");
	fclose(fp);
}

void refineMeshCPU_CGAL_without_acute_angles()
{
	int numofpoint, numoftri, numofedge = 0;
	int seed = 0;
	Distribution dist;
	double minarea = 0;
	tetgenio out;
	double radius_to_edge_ratio = 1.414;
	printf("Refining PLC using CGAL......\n");
	//for (radius_to_edge_ratio = 1.6; radius_to_edge_ratio <= 1.8; radius_to_edge_ratio += 0.2)
	{
		for (dist = (Distribution)2; dist <= (Distribution)2; dist = (Distribution)((int)dist + 1))
		{
			for (numofpoint = 25000; numofpoint <= 25000; numofpoint += 5000)
			{
				for (numoftri = numofpoint*0.05; numoftri <= numofpoint*0.25; numoftri += numofpoint*0.05)
				{
					// Set up parameter

					printf("numofpoint = %d, numoftri = %d, numofedge = %d, seed = %d, distribution = %d, minareafactor = %f, radius-to-edge-radio = %lf\n",
						numofpoint, numoftri, numofedge, seed, dist, minarea, radius_to_edge_ratio);

					// Run refinement
					if (readResultFile_CGAL_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, radius_to_edge_ratio)) // already have result file
					{
						printf("Found result file!\n");
					}
					else if (readInputPLCFile_without_acute_angles(numofpoint, numoftri, numofedge, seed, dist, minarea, &out))
					{
						printf("Refinement in process......\n");
						time_t rawtime;
						struct tm * timeinfo;
						time(&rawtime);
						timeinfo = localtime(&rawtime);
						printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

						// Prepare common line
						char *fname;
						{
							std::ostringstream strs;
							strs << "input_off_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
								<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge << ".off";
							std::string fn = strs.str();
							fname = new char[fn.length() + 1];
							strcpy(fname, fn.c_str());
						}

						std::cout.precision(17);
						std::cerr.precision(17);
						CGAL::Timer t;
						t.start();

						std::ifstream input(fname);
						const char* fname2 = "input_off_without_acute_angle/bounding_box.off";
						std::ifstream input2(fname2);
						Polyhedron sm, smbounding;
						input >> sm;
						input2 >> smbounding;
						if (input.fail()) {
							std::cerr << "Error: Cannot read file " << fname << std::endl;
							continue;
						}

						// Create domain
						Mesh_domain domain(sm, smbounding);

						// Add input points as corners
						std::vector<K::Point_3> points;
						for (int i = 8; i < out.numberofpoints; i++)
						{
							points.push_back(
								K::Point_3(out.pointlist[3 * i], out.pointlist[3 * i + 1], out.pointlist[3 * i + 2]));
						}
						domain.add_corners(points.begin(), points.end());

						tetgenio::facet *f, *f1;
						tetgenio::polygon *p, *p1;
						std::vector<std::vector<K::Point_3>> segments;
						// Add segments of the bounding box
						f = &out.facetlist[0];
						p = &f->polygonlist[0];
						f1 = &out.facetlist[1];
						p1 = &f1->polygonlist[0];
						for (int i = 0; i < 4; i++)
						{
							int v[4];
							v[0] = p->vertexlist[i];
							v[1] = p->vertexlist[(i + 1) % 4];
							v[2] = p1->vertexlist[i];
							v[3] = p1->vertexlist[(i + 1) % 4];
							std::vector<K::Point_3> segment0, segment1, segment2;
							segment0.push_back(
								K::Point_3(out.pointlist[3 * v[0]], out.pointlist[3 * v[0] + 1], out.pointlist[3 * v[0] + 2]));
							segment0.push_back(
								K::Point_3(out.pointlist[3 * v[1]], out.pointlist[3 * v[1] + 1], out.pointlist[3 * v[1] + 2]));
							segment1.push_back(
								K::Point_3(out.pointlist[3 * v[2]], out.pointlist[3 * v[2] + 1], out.pointlist[3 * v[2] + 2]));
							segment1.push_back(
								K::Point_3(out.pointlist[3 * v[3]], out.pointlist[3 * v[3] + 1], out.pointlist[3 * v[3] + 2]));
							segment2.push_back(
								K::Point_3(out.pointlist[3 * v[0]], out.pointlist[3 * v[0] + 1], out.pointlist[3 * v[0] + 2]));
							segment2.push_back(
								K::Point_3(out.pointlist[3 * v[2]], out.pointlist[3 * v[2] + 1], out.pointlist[3 * v[2] + 2]));
							segments.push_back(segment0);
							segments.push_back(segment1);
							segments.push_back(segment2);
						}

						// Add segments of the inside subfaces
						for (int i = 6; i < out.numberoffacets; i++)
						{
							f = &out.facetlist[i];
							p = &f->polygonlist[0];
							for (int j = 0; j < 4; j++)
							{
								int v[2];
								v[0] = p->vertexlist[j];
								v[1] = p->vertexlist[(j + 1) % 4];
								std::vector<K::Point_3> segment;
								segment.push_back(
									K::Point_3(out.pointlist[3 * v[0]], out.pointlist[3 * v[0] + 1], out.pointlist[3 * v[0] + 2]));
								segment.push_back(
									K::Point_3(out.pointlist[3 * v[1]], out.pointlist[3 * v[1] + 1], out.pointlist[3 * v[1] + 2]));
								segments.push_back(segment);
							}
						}

						domain.add_features(segments.begin(), segments.end());

						// Mesh criteria
						double criteria_edge_size;
						double criteria_facet_angle, criteria_facet_size, criteria_facet_distance;
						double criteria_cell_size;
						criteria_edge_size = 5;
						criteria_facet_angle = 25;
						criteria_facet_distance = 1;
						Mesh_criteria criteria(
							edge_size = criteria_edge_size,
							/*facet_angle = criteria_facet_angle,*/ facet_distance = criteria_facet_distance, //facet_size = 5.0,
							cell_radius_edge_ratio = radius_to_edge_ratio/*, cell_size = 10.0*/);

						// Mesh generation
						C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria,
							no_perturb(), no_exude());

						//debug_CGAL_output(c3t3, radius_to_edge_ratio);

						double cpu_time = t.time() * 1000;
						int number_of_vertices = c3t3.triangulation().number_of_vertices();
						int number_of_facets = c3t3.triangulation().number_of_finite_facets();
						int number_of_edges = c3t3.triangulation().number_of_finite_edges();
						int number_of_cells = c3t3.triangulation().number_of_finite_cells();

						//std::ofstream medit_file("out_1.mesh");
						//c3t3.output_to_medit(medit_file);

						std::map<Tr3::Vertex_handle, int> vertices;
						double* pointlist = new double[3 * number_of_vertices];
						int* tetlist = new int[4 * number_of_cells];
						int index = 0;
						for (
							Tr3::Finite_vertices_iterator vit = c3t3.triangulation().finite_vertices_begin(),
							end = c3t3.triangulation().finite_vertices_end(); vit != end; ++vit)
						{
							vertices[vit] = index;
							pointlist[3 * index + 0] = vit->point().x();
							pointlist[3 * index + 1] = vit->point().y();
							pointlist[3 * index + 2] = vit->point().z();
							index++;
						}

						index = 0;
						for (
							Tr3::Finite_cells_iterator cit = c3t3.triangulation().finite_cells_begin(),
							end = c3t3.triangulation().finite_cells_end(); cit != end; ++cit)
						{
							for (int i = 0; i < 4; i++)
							{
								int v = vertices.find(cit->vertex(i))->second;
								tetlist[4 * index + i] = v;
							}
							index++;
						}

						int numofbadtets = countBadTets(pointlist, tetlist, number_of_cells, radius_to_edge_ratio);

						printf("Number of points = %d\n", number_of_vertices);
						printf("Number of subfaces = %d\n", number_of_facets);
						printf("Number of segments = %d\n", number_of_edges);
						printf("Number of tetrahedra = %d\n", number_of_cells);
						printf("Number of bad tetrahedra = %d\n", numofbadtets);
						printf("Total time = %lf\n", cpu_time);
						printf("  edge_size = %lf\n", criteria_edge_size);
						//printf("  facet_angle = %lf\n", criteria_facet_angle);
						printf("  facet_distance = %lf\n", criteria_facet_distance);
						printf("  cell_radius_edge_ratio = %lf\n", radius_to_edge_ratio);

						// Prepare for fileName
						char *fileName;
						{
							std::ostringstream strs;
							strs << "result_without_acute_angle/d" << dist << "_s" << seed << "_a" << minarea
								<< "_p" << numofpoint << "_t" << numoftri << "_e" << numofedge
								<< "_q" << radius_to_edge_ratio << "_cgal" << ".txt";
							std::string fntmp = strs.str();
							fileName = new char[fntmp.length() + 1];
							strcpy(fileName, fntmp.c_str());
						}

						// Save information into files
						FILE * fp;
						fp = fopen(fileName, "w");
						fprintf(fp, "Number of points = %d\n", number_of_vertices);
						fprintf(fp, "Number of subfaces = %d\n", number_of_facets);
						fprintf(fp, "Number of segments = %d\n", number_of_edges);
						fprintf(fp, "Number of tetrahedra = %d\n", number_of_cells);
						fprintf(fp, "Number of bad tetrahedra = %d\n", numofbadtets);
						fprintf(fp, "Total time = %lf\n", cpu_time);
						fprintf(fp, "  edge_size = %lf\n", criteria_edge_size);
						fprintf(fp, "  facet_distance = %lf\n", criteria_facet_distance);
						//fprintf(fp, "  facet_angle = %lf\n", criteria_facet_angle);
						fprintf(fp, "  cell_radius_edge_ratio = %lf\n", radius_to_edge_ratio);
						fclose(fp);
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