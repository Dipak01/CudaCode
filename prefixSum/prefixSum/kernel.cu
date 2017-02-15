/*
INPUT: An integer n and an array A = (a_1, ..., a_n) of floating point numbers .

OUTPUT : An array of legth n : (a_1, 2 * a_1 + a_2, 3 * a_1 + 2 * a_2 + a_3, 4 * a_1 + 3 * a_2 + 2 * a_3 + a_4, ...)
That is, the ith member of your array must be i*a_1 + (i - 1)*a_2 + ... + 2 * a_{ i - 1 } +a_i.

Design a parallel algorithm that runs in TIME(log n) and implement it on the CUDA platform.

Example input : (2, 0, 7)
Example output : (2, 4, 13)
*/

#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD 1024

using namespace std;

// Kernel function to generate sequence
__global__
void add(int n, float *x, int outerLoopIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		int k = (1 << outerLoopIdx) + 1;
		
		if (i - k + 1 >= 0)
			x[i] = x[i - k + 1] + x[i];	
	}
}

int main(void)
{
	int n = 1024;

	float *x, *d_x;

	//Allocate memory on CPU
	x = (float*)malloc(n * sizeof(float));

	//Allocate memory on GPU
	cudaMalloc(&d_x, n * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 0.0f + i;
	}

	//Copy memory from CPU to GPU 
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

	clock_t begin, end;
	begin = clock();

	//Perform prefix sum
	for (int k = 0; k < 2; k++){
		for (int i = 0; i < 10; i++){
			add << <n / THREAD + (n % THREAD != 0), THREAD >> >(n, d_x, i);
		}
	}
	
	end = clock();
	int time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "The running time for parallel addition is ";
	cout << time_spent << " miliseconds." << endl;


	//Copy memory from GPU to CPU 
	cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

	//Print Results
	for (int i = 0; i < n; i++){
		cout << x[i] << ", ";
	}

	//Free memory on GPU
	cudaFree(d_x);

	//Free memory on CPU
	free(x);

	return 0;
}