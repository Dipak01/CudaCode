#include "genresult.cuh"
#include <sys/time.h>

//kernel code same as in problem statement
__global__ void getMulAtomic_kernel(const int nz, const int * coord_row, const int * coord_col, const float * vec, const float * mat, float * res){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int totalThread = blockDim.x * gridDim.x;
	int iter = (nz % totalThread) ? (nz / totalThread + 1) : (nz / totalThread);

	for (int i = 0; i < iter; i++) {
		int dataId = threadId + i * totalThread;
		if (dataId < nz) {
			float data = mat[dataId];
			int row = coord_row[dataId];
			int col = coord_col[dataId];
			float temp = data * vec[col];
			atomicAdd(&res[row], temp);
		}
	}
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	int entries = mat->nz, rows = mat->M, cols = mat->N;

	int *d_coord_row, *d_coord_col;
	float *d_vec, *d_mat, *d_res;

	//coord_col and coord_row have size set to nz
	cudaMalloc((void**)&d_coord_row, entries * sizeof(int));
	cudaMalloc((void**)&d_coord_col, entries * sizeof(int));
	cudaMalloc((void**)&d_mat, entries * sizeof(float));
	//M rows and N cols
	cudaMalloc((void**)&d_vec, cols * sizeof(float));
	cudaMalloc((void**)&d_res, rows * sizeof(float));
	//initially setting values to zero
	cudaMemset(d_res, 0, rows * sizeof(float));

	cudaMemcpy(d_coord_row, mat->rIndex, entries * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coord_col, mat->cIndex, entries * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat, mat->val, entries * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, vec->val, cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res->val, rows * sizeof(float), cudaMemcpyHostToDevice);
	
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	
	getMulAtomic_kernel <<<blockNum, blockSize>>>(entries, d_coord_row, d_coord_col, d_mat, d_vec, d_res);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	//copying result back
	cudaMemcpy(res->val, d_res, rows * sizeof(float), cudaMemcpyDeviceToHost);

	/*Deallocate.*/
	cudaFree(d_coord_col);
	cudaFree(d_coord_row);
	cudaFree(d_vec);
	cudaFree(d_mat);
	cudaFree(d_res);
}
