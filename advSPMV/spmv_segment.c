#include "genresult.cuh"
#include <sys/time.h>


//mergesort adapted from: http://www.cprogramming.com/tutorial/computersciencetheory/merge.html

//using struct data structure so as to ease reordering of elements
struct matStruct {
	int rNewIndex;
	int cNewIndex;
	float valNew;
};

matStruct max(matStruct x, matStruct y)
{
	if (x.rNewIndex < y.rNewIndex)
	{
		return x;
	}
	else
	{
		return y;
	}
}

void merge_helper(matStruct *input, int left, int right, matStruct *scratch)
{
	/* base case: one element */
	if (right == left + 1)
	{
		return;
	}
	else
	{
		int i = 0;
		int length = right - left;
		int midpoint_distance = length / 2;
		/* l and r are to the positions in the left and right subarrays */
		int l = left, r = left + midpoint_distance;

		/* sort each subarray */
		merge_helper(input, left, left + midpoint_distance, scratch);
		merge_helper(input, left + midpoint_distance, right, scratch);

		/* merge the arrays together using scratch for temporary storage */
		for (i = 0; i < length; i++)
		{
			/* Check to see if any elements remain in the left array; if so,
			* we check if there are any elements left in the right array; if
			* so, we compare them.  Otherwise, we know that the merge must
			* use take the element from the left array */
			if (l < left + midpoint_distance &&
				(r == right || max(input[l], input[r]).rNewIndex == input[l].rNewIndex))
			{
				scratch[i] = input[l];
				l++;
			}
			else
			{
				scratch[i] = input[r];
				r++;
			}
		}
		/* Copy the sorted subarray back to the input */
		for (i = left; i < right; i++)
		{
			input[i] = scratch[i - left];
		}
	}
}

void mergesort(matStruct *matData, int size)
{
	matStruct *scratch = (matStruct *)malloc(size * sizeof(matStruct));
	if (scratch != NULL)
	{
		merge_helper(matData, 0, size, scratch);
		free(scratch);
	}
}


void sortMatrix(MatrixInfo *mat){
	matStruct *matData = (matStruct *)malloc(mat->nz * sizeof(matStruct));

	int i;
	for (i = 0; i < mat->nz; i++){
		matData[i].rNewIndex = mat->rIndex[i];
		matData[i].cNewIndex = mat->cIndex[i];
		matData[i].valNew = mat->val[i];
	}

	mergesort(matData, mat->nz);

	for (i = 0; i < mat->nz; i++) {
		mat->rIndex[i] = matData[i].rNewIndex;
		mat->cIndex[i] = matData[i].cNewIndex;
		mat->val[i] = matData[i].valNew;
	}

}

__global__ void putProduct_kernel(const int nz, const int * coord_row, const int * coord_col, const float * mat, const float * vec, float * res) {
	__shared__ int rows[1024];
	__shared__ float vals[1024];

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThread = blockDim.x * gridDim.x;
	int iter = (nz % totalThread) ? (nz / totalThread + 1) : (nz / totalThread);

	for (int i = 0; i < iter; i++) {
		int dataId = threadId + i * totalThread;
		if (dataId < nz) {
			//save product into vals shared memory
			float data = mat[dataId];
			int col = coord_col[dataId];
			vals[threadIdx.x] = data * vec[col];
			//save row number into rows shared memory
			rows[threadIdx.x] = coord_row[dataId];
			//do prefix sum on vals array using the condition in rows array
			int lane = threadId % 32;
			if (lane >= 1 && rows[threadIdx.x] == rows[threadIdx.x - 1])
				vals[threadIdx.x] += vals[threadIdx.x - 1];
			if (lane >= 2 && rows[threadIdx.x] == rows[threadIdx.x - 2])
				vals[threadIdx.x] += vals[threadIdx.x - 2];
			if (lane >= 4 && rows[threadIdx.x] == rows[threadIdx.x - 4])
				vals[threadIdx.x] += vals[threadIdx.x - 4];
			if (lane >= 8 && rows[threadIdx.x] == rows[threadIdx.x - 8])
				vals[threadIdx.x] += vals[threadIdx.x - 8];
			if (lane >= 16 && rows[threadIdx.x] == rows[threadIdx.x - 16])
				vals[threadIdx.x] += vals[threadIdx.x - 16];

			//write output if we are dealing with last thread in warp or rows are different
			if ((lane == 31) || (rows[threadIdx.x] != rows[threadIdx.x + 1])){
				atomicAdd(&res[rows[threadIdx.x]], vals[threadIdx.x]);
			}
		}
	}
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	struct timespec start1, end1;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start1);
	sortMatrix(mat);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end1);
	printf("Reorganizing Time: %lu milli-seconds\n", 1000 * (end1.tv_sec - start1.tv_sec) + (end1.tv_nsec - start1.tv_nsec) / 1000000);

	int *d_coord_row, *d_coord_col;
	float *d_vec, *d_mat, *d_res;

	//coord_col and coord_row have size set to nz
	cudaMalloc((void**)&d_coord_row, mat->nz * sizeof(int));
	cudaMalloc((void**)&d_coord_col, mat->nz * sizeof(int));
	cudaMalloc((void**)&d_mat, mat->nz * sizeof(float));
	//M rows and N cols
	cudaMalloc((void**)&d_vec, mat->N * sizeof(float));
	cudaMalloc((void**)&d_res, mat->M * sizeof(float));
	//initially setting values to zero
	cudaMemset(d_res, 0, mat->M * sizeof(float));

	cudaMemcpy(d_coord_row, mat->rIndex, mat->nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coord_col, mat->cIndex, mat->nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat, mat->val, mat->nz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, vec->val, mat->N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res->val, mat->M * sizeof(float), cudaMemcpyHostToDevice);
    
	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	putProduct_kernel << <blockNum, blockSize >> >(mat->nz, d_coord_row, d_coord_col, d_mat, d_vec, d_res);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu milli-seconds\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);

	//copying result back
	cudaMemcpy(res->val, d_res, mat->M * sizeof(float), cudaMemcpyDeviceToHost);

	/*Deallocate.*/
	cudaFree(d_coord_col);
	cudaFree(d_coord_row);
	cudaFree(d_vec);
	cudaFree(d_mat);
	cudaFree(d_res);
}
