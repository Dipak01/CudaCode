//Explain the odd behavior of the following puzzle program 
//1. run it several times; 
//2. change the numbers 256, 256 to other pairs of integers

#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// Kernel function to increase a counter

__global__   
void count(int *x) { 
	*x = *x + 1; 
}

int main(void){

	int  *x, *d_x;

	//Allocate memory on CPU

	x = (int*)malloc(sizeof(int));

	//Allocate memory on GPU

	cudaMalloc(&d_x, sizeof(int));

	// initialize x on the host

	*x = 0;

	//Copy memory from CPU to GPU 

	cudaMemcpy(d_x, x, sizeof(int), cudaMemcpyHostToDevice);

	// Perform Counting on GPU

	count << < 16, 128 >> > (d_x);

	//Copy memory from GPU to CPU

	cudaMemcpy(x, d_x, sizeof(int), cudaMemcpyDeviceToHost);

	//Print Results

	printf("x is %d\n", *x);



	//Free memory on GPU
	cudaFree(d_x);
	//Free memory on CPU       
	free(x);

	return 0;

}