#include <iostream>
#include <cmath>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD 1024
#define POWER 25
#define k 16

using namespace std;

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
	int i = k*(blockIdx.x*blockDim.x + threadIdx.x);
	for (int j = i; j < i + k; j++) {
		if (j < n) y[j] = x[j] + y[j];
	}
}

int main(void)
{
	int n = (1 << POWER) - 7364;

	float *x, *y, *d_x, *d_y;
	cout << "Size of input: " << n << endl;

	unsigned int blocks = ((n + k*THREAD - 1) / (k*THREAD));
	cout << "Number of blocks: " << blocks << "\t\tNumber of threads per block: " << THREAD << "\t\tNumber of Indies per thread: " << k << endl;

	//Allocate memory on CPU
	x = (float*)malloc(n * sizeof(float));
	y = (float*)malloc(n * sizeof(float));

	//Allocate memory on GPU
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 3.0f;
		y[i] = 2.0f;
	}

	//Sequential add vectors
	clock_t begin = clock();
	for (int i = 0; i < n; i++) {
		y[i] = x[i] + y[i];
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "The running time for sequential addtition is " << time_spent << " miliseconds." << endl;

	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 3.0f;
		y[i] = 2.0f;
	}


	begin = clock();

	//Copy memory from CPU to GPU 
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

	// Perform Addition on GPU
	add << < blocks, THREAD >> >(n, d_x, d_y);

	//Copy memory from GPU to CPU 
	cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;

	//Verify Results
	bool isCorrect = true;
	for (int i = 0; i < n; i++){
		if (y[i] != 5.0) {
			cout << "Incorrect Result at " << i << " = " << y[i] << endl;
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) cout << "The running time for parallel addtition is " << time_spent << " miliseconds." << endl;

	//Free memory on GPU
	cudaFree(d_x);
	cudaFree(d_y);

	//Free memory on CPU
	free(x);
	free(y);

	return 0;
}