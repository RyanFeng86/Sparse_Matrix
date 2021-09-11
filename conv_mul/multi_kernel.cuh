#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>


template<class T>
__global__ void Mult_normal(T **cuda_filter, T **cuda_input, T **cuda_output, int filter_num, int filter_size, int channel_num, int input_hight, int input_width) {
	int bid = blockIdx.x;
	int locIdx = threadIdx.x;
	int globIdx = blockIdx.x*blockDim.x + threadIdx.x;

	//each block will creat one share memory strip for each row of filter array
	__shared__ T *filter;
	__shared__ T *result;
	filter = new T[blockDim.x];
	result = new T[blockDim.x];

	if (locIdx < filter_size*filter_size*channel_num) {
		filter[locIdx] = cuda_filter[bid][locIdx];
	}

	//__syncthreads();

	if (locIdx < blockDim.x) {
		for (int i = 0; i < input_hight*input_width; i++) {
			//printf(" -%d %d %d- ", i, bid,locIdx);
			//printf("\nxxxxxxxxxxxxxxxxxxx\n");
			result[locIdx] = filter[locIdx] * cuda_input[locIdx][i];
			//printf("%d ", result[locIdx]);
			__syncthreads();
			if (locIdx == 0) {
				T tmp = 0;
				for (int j = 0; j < blockDim.x; j++) {
					tmp += result[j];
					//printf("%d ", tmp);
				}
				cuda_output[bid][i] = tmp;
			}
			//__syncthreads();
		}
	}
}





/*
if (globIdx == 0) {
	printf("\n");
	printf("+++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < filter_size*filter_size*channel_num; j++) {
			printf("%d ", cuda_filter[i][j]);
		}
		printf("\n");
	}
	printf("-------------------------------\n");
	printf("\n");

	printf("\n");
	printf("+++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < filter_size*filter_size*channel_num; i++) {
		for (int j = 0; j < input_width*input_hight; j++) {
			printf("%d ", cuda_input[i][j]);
		}
		printf("\n");
	}
	printf("-------------------------------\n");
	printf("\n");

	printf("\n");
	printf("+++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < input_width*input_hight; j++) {
			printf("%d ", cuda_output[i][j]);
		}
		printf("\n");
	}
	printf("-------------------------------\n");
	printf("\n");
}


if (globIdx == 0) {
	printf("\n");
	printf("+++++++++++++++++++++++++++++++\n");
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < input_width*input_hight; j++) {
			printf("%d ", cuda_output[i][j]);
		}
		printf("\n");
	}
	printf("-------------------------------\n");
	printf("\n");
}
*/