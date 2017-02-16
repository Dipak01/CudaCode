#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//Note: following 2 header files added
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//Note: Global var declared
#define THREAD 1024

// Kernel function to generate sequence
__global__
void add(int n, double *x, int outerLoopIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		int k = (1 << outerLoopIdx) + 1;

		if (i - k + 1 >= 0)
			x[i] = x[i - k + 1] + x[i];
	}
}

void sum(double* a, double* b, const int n) {
	//Given an array a[0...n-1], you need to compute b[0...n-1],
	//where b[i] = (i+1)*a[0] + i*a[1] + ... + 2*a[i-1] + a[i]
	//Note that b is NOT initialized with 0, be careful!
	//Write your CUDA code starting from here
	//Add any functions (e.g., device function) you want within this file
	
	double *d_a;

	//Allocate memory on GPU
	cudaMalloc(&d_a, n * sizeof(double));

	//Copy memory from CPU to GPU 
	cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);

	for (int k = 0; k < 2; k++){
		for (int i = 0; i < log(n); i++){
			add << <n / THREAD + (n % THREAD != 0), THREAD >> >(n, d_a, i);
		}
	}

	//Copy memory from GPU to CPU 
	cudaMemcpy(a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

	return;
}

int main(int argc, const char * argv[]) {

	if (argc != 2) {
		printf("The argument is wrong! Execute your program with only input file name!\n");
		return 1;
	}

	//Dummy code for creating a random input vectors
	//Convenient for the text purpose
	//Please comment out when you submit your code!!!!!!!!! 	
	FILE *fp = fopen(argv[1], "w");
	if (fp == NULL) {
	printf("The file can not be created!\n");
	return 1;
	}
	int n = 1 << 24;
	fprintf(fp, "%d\n", n);
	srand(time(NULL));
	for (int i=0; i<n; i++)
	fprintf(fp, "%lg\n", ((double)(rand() % n))/100);
	fclose(fp);
	printf("Finished writing\n");

	//Read input from input file specified by user
	//Note: Uncomment line below before submission
	//FILE* fp = fopen(argv[1], "r"); 
	fp = fopen(argv[1], "r");
	if (fp == NULL) {
		printf("The file can not be opened or does not exist!\n");
		return 1;
	}
	//Note: Uncomment line below before submission
	//int n;
	fscanf(fp, "%d\n", &n);
	printf("%d\n", n);
	//Note: Typecasrting done as it was giving a compile time error
	double* a = (double*)malloc(n*sizeof(double));
	double* b = (double*)malloc(n*sizeof(double));
	for (int i = 0; i<n; i++) {
		fscanf(fp, "%lg\n", &a[i]);
	}
	fclose(fp);

	//Main function
	sum(a, b, n);

	//Write b into output file
	fp = fopen("output.txt", "w");
	if (fp == NULL) {
		printf("The file can not be created!\n");
		return 1;
	}
	fprintf(fp, "%d\n", n);
	//Note: Changed to 'a' from 'b' as we can manage without 'b'
	for (int i = 0; i<n; i++)
		fprintf(fp, "%lg\n", a[i]);
	fclose(fp);
	free(a);
	free(b);
	printf("Done...\n");
	return 0;
}