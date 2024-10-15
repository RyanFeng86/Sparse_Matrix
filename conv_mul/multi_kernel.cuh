
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>




template<class T>
 __global__ void Mult_normal(T **cuda_filter, T **cuda_input, T **cuda_output,int *cuda_input_hight, int *cuda_input_width) {
	int bid = blockIdx.x;
	int locIdx = threadIdx.x;
	//int globIdx = blockIdx.x*blockDim.x + threadIdx.x;

	//each block will creat one share memory strip for each row of filter array
	__shared__ T *filter;
	__shared__ T *result;
	filter = new T[blockDim.x];
	result = new T[blockDim.x];

	if (locIdx < blockDim.x) {
		filter[locIdx] = cuda_filter[bid][locIdx];
	}

	//__syncthreads();


	if (locIdx < blockDim.x) {
		for (int i = 0; i < (*cuda_input_hight)*(*cuda_input_width); i++) {
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
	delete[] filter;
	delete[] result;
	
}


 template<class T>
 __global__ void Mult_improve(T **cuda_filter, T **cuda_index, T **cuda_input, T **cuda_output, int *cuda_input_hight, int *cuda_input_width, int * cuda_divider, int *cuda_each_group) {
	 int bid = blockIdx.x;
	 int locIdx = threadIdx.x;
	 //int globIdx = blockIdx.x*blockDim.x + threadIdx.x;

	 //each block will creat one share memory strip for each row of filter array
	 __shared__ T *filter;
	 __shared__ T *result;
	 __shared__ int input_hight;
	 __shared__ int input_width;
	 __shared__ int each_group;
	 __shared__ int divider;
	 if (locIdx == 0){
		 filter = new T[blockDim.x];
		 result = new T[blockDim.x];
		 input_hight = (*cuda_input_hight);
		 input_width = (*cuda_input_width);
		 each_group = (*cuda_each_group);
		 divider = (*cuda_divider);
	 }
	 __syncthreads();
	 int index_;
	 if (locIdx < blockDim.x) {
		 index_ = (locIdx / each_group)*divider + cuda_index[bid][locIdx];
		 filter[locIdx] = cuda_filter[bid][index_];		 
	 }

	 //__syncthreads();


	 if (locIdx < blockDim.x) {
		 for (int i = 0; i < input_hight*input_width; i++) {
			 //printf(" -%d %d %d- ", i, bid,locIdx);
			 //printf("\nxxxxxxxxxxxxxxxxxxx\n");
			 result[locIdx] = filter[locIdx] * cuda_input[index_][i];
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