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
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] = x[i] + y[i];
	}
}

int main(void)
{
	int n = 1 << POWER;

	float *x, *y, *d_x, *d_y;

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

	//Copy memory from CPU to GPU 
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

	//Sequential add vectors
	clock_t begin = clock();
	for (int i = 0; i < n; i++) {
		y[i] = x[i] + y[i];
	}
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "The running time for sequential addtition is " << time_spent << " miliseconds." << endl;


	begin = clock();
	// Perform Addition on GPU

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
		if (y[i] != 5.0) {
			cout << "Incorrect Result" << endl;
			flag = false;
			break;
		}
	}
	if (flag) cout << "Correct! Welcome to CUDA world!" << endl;

	//Free memory on GPU
	cudaFree(d_x);
	cudaFree(d_y);

	//Free memory on CPU
	free(x);
	free(y);

	return 0;
}