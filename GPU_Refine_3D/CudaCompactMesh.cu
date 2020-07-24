#include "CudaCompactMesh.h"
#include "CudaMesh.h"

void compactMesh(
	int& out_numofpoint,
	double*& out_pointlist,
	RealD& t_pointlist,
	int& out_numofedge,
	int*& out_edgelist,
	IntD& t_seglist,
	TriStatusD& t_segstatus,
	int& out_numoftriface,
	int*& out_trifacelist,
	IntD& t_trifacelist,
	TriStatusD& t_tristatus,
	int& out_numoftet,
	int*& out_tetlist,
	IntD& t_tetlist,
	TetStatusD& t_tetstatus
)
{
	IntD t_sizes, t_indices, t_list;
	int numberofthreads, numberofblocks;

	out_numofpoint = t_pointlist.size() / 3;
	out_pointlist = new double[3 * out_numofpoint];
	cudaMemcpy(out_pointlist, thrust::raw_pointer_cast(&t_pointlist[0]), 3 * out_numofpoint * sizeof(double), cudaMemcpyDeviceToHost);

	int last_edge = t_segstatus.size();
	t_sizes.resize(last_edge);
	t_indices.resize(last_edge);
	thrust::fill(t_sizes.begin(), t_sizes.end(), 1);
	thrust::replace_if(t_sizes.begin(), t_sizes.end(), t_segstatus.begin(), isEmptyTri(), 0);
	thrust::exclusive_scan(t_sizes.begin(), t_sizes.end(), t_indices.begin());
	out_numofedge = thrust::reduce(t_sizes.begin(), t_sizes.end());
	out_edgelist = new int[2 * out_numofedge];
	t_list.resize(2 * out_numofedge);

	numberofthreads = last_edge;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCompactSeg << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_seglist[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		thrust::raw_pointer_cast(&t_indices[0]),
		thrust::raw_pointer_cast(&t_list[0]),
		numberofthreads
		);
	cudaMemcpy(out_edgelist, thrust::raw_pointer_cast(&t_list[0]), 2 * out_numofedge * sizeof(int), cudaMemcpyDeviceToHost);

	int last_triface = t_tristatus.size();
	t_sizes.resize(last_triface);
	t_indices.resize(last_triface);
	thrust::fill(t_sizes.begin(), t_sizes.end(), 1);
	thrust::replace_if(t_sizes.begin(), t_sizes.end(), t_tristatus.begin(), isEmptyTri(), 0);
	thrust::exclusive_scan(t_sizes.begin(), t_sizes.end(), t_indices.begin());
	out_numoftriface = thrust::reduce(t_sizes.begin(), t_sizes.end());
	out_trifacelist = new int[3 * out_numoftriface];
	t_list.resize(3 * out_numoftriface);

	numberofthreads = last_triface;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);
	kernelCompactTriface << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_trifacelist[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		thrust::raw_pointer_cast(&t_indices[0]),
		thrust::raw_pointer_cast(&t_list[0]),
		numberofthreads
		);
	cudaMemcpy(out_trifacelist, thrust::raw_pointer_cast(&t_list[0]), 3 * out_numoftriface * sizeof(int), cudaMemcpyDeviceToHost);

	int last_tet = t_tetstatus.size();
	t_sizes.resize(last_tet);
	t_indices.resize(last_tet);
	thrust::fill(t_sizes.begin(), t_sizes.end(), 1);
	numberofthreads = last_tet;
	numberofblocks = (ceil)((float)numberofthreads / BLOCK_SIZE);

	kernelCompactTet_Phase1 << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_tetstatus[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		numberofthreads
		);

	thrust::exclusive_scan(t_sizes.begin(), t_sizes.end(), t_indices.begin());
	out_numoftet = thrust::reduce(t_sizes.begin(), t_sizes.end());
	out_tetlist = new int[4 * out_numoftet];
	t_list.resize(4 * out_numoftet);

	kernelCompactTet_Phase2 << <numberofblocks, BLOCK_SIZE >> > (
		thrust::raw_pointer_cast(&t_tetlist[0]),
		thrust::raw_pointer_cast(&t_sizes[0]),
		thrust::raw_pointer_cast(&t_indices[0]),
		thrust::raw_pointer_cast(&t_list[0]),
		numberofthreads
		);
	cudaMemcpy(out_tetlist, thrust::raw_pointer_cast(&t_list[0]), 4 * out_numoftet * sizeof(int), cudaMemcpyDeviceToHost);
}