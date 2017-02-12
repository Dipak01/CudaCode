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
#define POWER 25

using namespace std;

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
	//blockID*blockDim + threadID
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//threads are 16 but vectors are only 15. So to handle this extra thread we added this check.
	if (i < n) {
		for (int j = i; j >= 0; j--) {
			y[i] = y[i] + (j + 1) * x[i - j];
		}
	}
}

int main(void)
{
	int n = 1024; //1 << POWER; //left shift or 2^Power

	float *x, *y, *d_x, *d_y;

	//Allocate memory on CPU
	x = (float*)malloc(n * sizeof(float));
	y = (float*)malloc(n * sizeof(float));

	//Allocate memory on GPU
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 0.0f + i;
		y[i] = 0.0f;
	}

	//Copy memory from CPU to GPU 
	//destination, source, size of mem we need to copy, direction
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

	//Sequential add vectors
	clock_t begin = clock();
	for (int i = 0; i < n; i++) {
		for (int j = i; j >= 0; j--) {
	//		y[i] = y[i] + (j + 1) * x[i - j];
		}		
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "The running time for sequential addtition is " << time_spent << " miliseconds." << endl;


	begin = clock();

	// Perform Addition on GPU
	//ceiling done as default there is floor and 1 block of threads will be ignored.
	//Alternate thing: n/thread + (n % thread != 0)
	add << <(n + THREAD - 1) / THREAD, THREAD >> >(n, d_x, d_y);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "The running time for parallel addition is ";
	cout << time_spent << " miliseconds." << endl;


	//Copy memory from GPU to CPU 
	cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);


	bool flag = true;
	//Verify Results
	for (int i = 0; i < n; i++){
		cout << y[i] <<", ";
	}
	//if (flag) cout << "Correct! Welcome to CUDA world!" << endl;

	//Free memory on GPU
	cudaFree(d_x);
	cudaFree(d_y);

	//Free memory on CPU
	free(x);
	free(y);

	return 0;
}