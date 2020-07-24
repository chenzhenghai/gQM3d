#include "CudaThrust.h"
#include "thrust/execution_policy.h"

int updateActiveListByMarker_Slot
(
	IntD	    &t_marker,
	IntD		&t_active,
	int         numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			isNotNegativeInt()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByMarker_Slot
(
	ByteD	    &t_marker,
	IntD		&t_active,
	int         numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			isNotNegativeByte()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByStatus_Val
(
	IntD		&t_input,
	TriStatusD	&t_status,
	IntD		&t_active,
	int			numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			t_input.begin(),
			t_input.end(),
			t_status.begin(),
			t_active.begin(),
			isNotEmptyTri()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByStatus_Slot
(
	TetStatusD	&t_status,
	IntD		&t_active,
	int			numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_status.begin(),
			t_active.begin(),
			isBadTet()),
		t_active.end());

	return t_active.size();
}

int updateEmptyTets
(
	TetStatusD	&t_tetstatus,
	IntD	&t_emptytets
)
{
	const int tetlistsize = t_tetstatus.size();

	t_emptytets.resize(tetlistsize);

	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last(tetlistsize);

	t_emptytets.erase(
		thrust::copy_if(
			first,
			last,
			t_tetstatus.begin(),
			t_emptytets.begin(),
			isEmptyTet()),
		t_emptytets.end());

	return t_emptytets.size();
}

int updateEmptyTris
(
	TriStatusD &t_tristatus,
	IntD	&t_emptytris
)
{
	const int trilistsize = t_tristatus.size();

	t_emptytris.resize(trilistsize);

	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last(trilistsize);

	t_emptytris.erase(
		thrust::copy_if(
			first,
			last,
			t_tristatus.begin(),
			t_emptytris.begin(),
			isEmptyTri()),
		t_emptytris.end());

	return t_emptytris.size();
}

void gpuMemoryCheck()
{
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if (cudaSuccess != cuda_status)
	{
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}