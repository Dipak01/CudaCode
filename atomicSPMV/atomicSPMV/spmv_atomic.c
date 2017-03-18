#include "genresult.cuh"
#include <sys/time.h>
//http://berenger.eu/blog/cusparse-cccuda-sparse-matrix-examples-csr-bcsr-spmv-and-conversions/

__global__ void getMulAtomic_kernel(const int nz, const int * coord_row, const int * coord_col, const float * vec, const float * mat, const float * res){
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
    /*Allocate here...*/
	int *d_coord_row, *d_coord_col;
	float *d_vec, *d_mat, *d_res;

	//in mmio rIndex = (int *)malloc(nz * sizeof(int));
	//so size has been set to nz*int
	//similar for coord_col
	cudaMalloc(&d_coord_row, mat->nz * sizeof(int));
	cudaMalloc(&d_coord_col, mat->nz * sizeof(int));
	cudaMalloc(&d_mat, mat->nz * sizeof(float));
	cudaMalloc(&d_vec, vec->nz * sizeof(float));
	cudaMalloc(&d_res, res->nz * sizeof(float));
	
	cudaMemset(d_res, 0, res->nz * sizeof(float));

	cudaMemcpy(d_coord_row, mat->rIndex, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coord_col, mat->cIndex, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat, mat->val, mat->nz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, vec->val, vec->nz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res->val, res->nz * sizeof(float), cudaMemcpyHostToDevice);

	/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/
	getMulAtomic_kernel <<<blockNum, blockSize >>>(mat->nz, d_coord_row, d_coord_col, d_mat, d_vec, d_res);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    /*please modify the timing to milli-seconds*/
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
 
	//copying result back
	cudaMemcpy(res->val, d_res, res->nz * sizeof(float), cudaMemcpyDeviceToHost);

	/*Deallocate.*/
	cudaFree(d_coord_col);
	cudaFree(d_coord_row);
	cudaFree(d_vec);
	cudaFree(d_mat);
	cudaFree(d_res);
}
