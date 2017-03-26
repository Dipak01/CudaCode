#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulDesign_kernel(const int nz, const int * coord_row, const int * coord_col, const float * mat, const float * vec, float * res){
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

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	sortMatrix(mat);

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

	getMulAtomic_kernel << <blockNum, blockSize >> >(mat->nz, d_coord_row, d_coord_col, d_mat, d_vec, d_res);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	//copying result back
	cudaMemcpy(res->val, d_res, mat->M * sizeof(float), cudaMemcpyDeviceToHost);

	/*Deallocate.*/
	cudaFree(d_coord_col);
	cudaFree(d_coord_row);
	cudaFree(d_vec);
	cudaFree(d_mat);
	cudaFree(d_res);
}
