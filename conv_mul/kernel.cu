#include <iostream>
#include <string>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
//CUDA RunTime API
#include <cuda_runtime.h>
#include <cuda.h>
#include "multi_kernel.cuh"

using namespace std;


enum sparsity { one_four, two_four, one_eight };
enum cal_model { normal_GPU, improve_GPU };


template<class T>
class con_mul{
public:
	con_mul(int a, int b, int c, int d, int e, sparsity y, cal_model z);
	~con_mul() {}

	bool gpu_info();
	void matrix_generation();
	void matrix_transform();
	
	void kernel_CPU();
	void kernel_GPU();
	void output_transform();
	void deal_with_time();

private:
	int filter_size;
	int channel_num;
	int filter_num;
	int input_hight;
	int input_width;
	sparsity matrix_sparsity;
	cal_model model;

	T ****filter_array_tmp;

	int divider;
	int **index;

	T **filter_array;
	T **input_array;

	T **output_tmp;
	T ***output;


	vector<double> CPU_time;
	vector<float> GPU_time;

	T **cuda_filter;
	T **cuda_input;
	T **cuda_output;

	cudaEvent_t start, stop;
};

template<class T>
con_mul<T>::con_mul(int a, int b, int c, int d, int e, sparsity y, cal_model z) {
	//init seed
	srand(time(0));

	//init filter, input size and model
	filter_size = a;
	channel_num = b;
	filter_num = c;
	input_hight = d;
	input_width = e;
	matrix_sparsity = y;
	model = z;

	//init filter_array_tmp
	filter_array_tmp = new T ***[filter_num];
	for (int i = 0; i < filter_num; i++)
		filter_array_tmp[i] = new T **[channel_num];
	for (int i = 0; i < filter_num; i++)
		for (int j = 0; j < channel_num; j++)
			filter_array_tmp[i][j] = new T *[filter_size];
	for (int i = 0; i < filter_num; i++)
		for (int j = 0; j < channel_num; j++)
			for (int k = 0; k < filter_size; k++)
				filter_array_tmp[i][j][k] = new T [filter_size];
	
	//init divider
	switch (matrix_sparsity) {
	case one_four:
		divider = 4;
		break;
	case one_eight:
		divider = 8;
		break;
	case two_four:
		divider = 2;
		break;
	default:
		break;
	}

	//init index
	index = new int *[filter_num];
	for (int i = 0; i < filter_num; i++)
		index[i] = new int [filter_size*filter_size*channel_num / divider];

	//init filter_array
	filter_array = new T *[filter_num];
	for (int i = 0; i < filter_num; i++)
		filter_array[i] = new T [filter_size*filter_size*channel_num];

	//init input_array
	input_array = new T *[filter_size*filter_size*channel_num];
	for (int i = 0; i < filter_size*filter_size*channel_num; i++)
		input_array[i] = new T[input_hight*input_width];

	//init output_tmp
	output_tmp = new T *[filter_num];
	for (int i = 0; i < filter_num; i++)
		output_tmp[i] = new T [input_hight*input_width];

	//init output
	output = new T **[filter_num];
	for (int i = 0; i < filter_num; i++)
		output[i] = new T *[input_hight];
	for (int i = 0; i < filter_num; i++)
		for (int j = 0; j < input_hight; j++)
			output[i][j] = new T [input_width];

	

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

}

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name); //device ASCII name
	//printf("totalGlobalMem : %ld.\n", prop.totalGlobalMem); //Total available  memoery in Byte
	cout <<"totalGlobalMem: "<< prop.totalGlobalMem/(1024*1024*1024)<<" GB" << endl;
	//printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock); //maximum memory for each thread on GPU
	cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock / (1024) << " KB" << endl;
	//printf("totalConstMem : %d.\n", prop.totalConstMem); //total const memory 
	cout << "totalConstMem: " << prop.totalConstMem / (1024) << " KB" << endl;
	printf("regsPerBlock : %d.\n", prop.regsPerBlock); //maximum 32bit register number for block
	printf("warpSize : %d.\n", prop.warpSize); //warp size
	printf("memPitch : %d.\n", prop.memPitch); // the farest distance for cudaMalloc 
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock); //maximum thread number in each block
	printf("maxBlocksPerMultiProcessor : %d.\n", prop.maxBlocksPerMultiProcessor);//maximum block number in each processor
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); //maximum value for each thread dimension
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); //maximum value for each grid dimension
	
	printf("major.minor : %d.%d.\n", prop.major, prop.minor); //major and minor number
	printf("clockRate : %d.\n", prop.clockRate); // clock rate in KHz
	printf("textureAlignment : %d.\n", prop.textureAlignment); //
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount); //processors number on the device
}
template<class T>
bool con_mul<T>::gpu_info() {
	int count;
	//get cuda support device number
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//print device info
		printDeviceProp(prop);
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}
	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);//Set it as working GPU
	return true;
}

template<class T>
void con_mul<T>::matrix_generation() {
	//generate filter_array_tmp
	cout << "*********************************" << endl;
	cout << "filter numer:" <<" "<<filter_num<< endl;
	cout << "filter channel number:" << " " << channel_num << endl;
	cout << "filter high size:" << " " << filter_size << endl;
	cout << "filter wide size:" << " " << filter_size << endl;
	for (int i = 0; i < filter_num; i++)
		for (int j = 0; j < channel_num; j++)
			for (int m = 0; m < filter_size; m++)
				for (int n = 0; n < filter_size; n++)
					filter_array_tmp[i][j][m][n]= (T)(rand() / double(RAND_MAX) + rand() % 10);
	
	
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < channel_num; j++) {
			for (int m = 0; m < filter_size; m++) {
				for (int n = 0; n < filter_size; n++) {
					cout << filter_array_tmp[i][j][m][n] << " ";
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}

	//generate index	
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < filter_size*filter_size*channel_num / divider; j++) {
			if (divider == 4)
				index[i][j] = rand() % 4;
			if (divider == 8)
				index[i][j] = rand() % 8;
			if (divider == 2) {				
				index[i][j] = rand() % 3;
				if (index[i][j] == 2)
					index[i][j + 1] = 3;
				else
					do {
						index[i][j + 1] = rand() % (3 - index[i][j]) + index[i][j];
					} while (index[i][j + 1] == index[i][j]);
				j++;
			}
		}
	}
	

	//generate input_array
	cout << endl;
	cout << "The following are input array" << endl;
	cout << "input high size:" << " " << input_hight << endl;
	cout << "input wide size:" << " " << input_width << endl;
	cout << "input after tranform high size: " << filter_size * filter_size*channel_num << endl;
	cout << "input after tranform width size " << input_hight * input_width << endl;
	for (int i = 0; i < filter_size*filter_size*channel_num; i++)
		for (int j = 0; j < input_hight*input_width; j++)
			input_array[i][j] = (T)(rand() / double(RAND_MAX) + rand() % 10);

	for (int i = 0; i < filter_size*filter_size*channel_num; i++) {
		for (int j = 0; j < input_hight*input_width; j++) {
			cout << input_array[i][j] << " ";
		}
		cout << endl;
	}	
}

template<class T>
void con_mul<T>::matrix_transform() {
	//transform filter_array_tmp to filter_array
	cout << "****************" << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < channel_num; j++) {
			for (int m = 0; m < filter_size; m++) {
				for (int n = 0; n < filter_size; n++) {
					filter_array[i][j*filter_size*filter_size + m * filter_size + n] =
						filter_array_tmp[i][j][m][n];
				}
			}
		}
	}

	cout << endl;
	cout << "Filter Array high size: " << filter_num << endl;
	cout << "Filter Array width size: " << filter_size * filter_size*channel_num << endl;
	cout << "Filter Array before insert 0 is shown as follows" << endl;	
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < filter_size*filter_size*channel_num; j++) {
			cout << filter_array[i][j] << " ";
		}
		cout << endl;
	}

	//
	cout << endl;
	cout << "the following outputs are index" << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < filter_size*filter_size*channel_num / divider; j++)
			cout << index[i][j] << " ";
		cout << endl;
	}
	cout << endl;



	//insert sparsity
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < filter_size*filter_size*channel_num/divider; j++) {
			int tmp = 0;
			if (divider == 2) {
				tmp = (j / 2) * 4 + index[i][j];
			}
			else {
				tmp = j * divider + index[i][j];
			}
			filter_array[i][tmp] = 0;
		}
	}


	cout << endl;
	cout << "Filter Array is shown as follows" << endl;
	cout << "Filter Array high size: " << filter_num << endl;
	cout << "Filter Array width size: " << filter_size * filter_size*channel_num << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < filter_size*filter_size*channel_num; j++) {
			cout << filter_array[i][j] << " ";
		}
		cout << endl;
	}
}

template<class T>
void con_mul<T>::kernel_CPU() {
	
	for (int i = 0; i < filter_num; i++)
		for (int j = 0; j <input_hight*input_width; j++)
			output_tmp[i][j]=0;

	double k = GetTickCount();
	for (int k = 0; k < input_hight*input_width; k++) {
		for (int i = 0; i < filter_num; i++) {
			for (int j = 0; j < filter_size*filter_size*channel_num; j++) {
				output_tmp[i][k] += filter_array[i][j] * input_array[j][k];				
			}
		}		
	}
	CPU_time.push_back((GetTickCount() - k)*1000);
	
	//display output tmp
	cout << endl;
	cout << "*******************" << endl;
	cout << "The following are CPU results" << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < input_hight*input_width; j++) {
			cout << output_tmp[i][j] << " ";
		}
		cout << endl;
	}
	

}





template<class T>
void con_mul<T>::kernel_GPU() {
	
	if (model == normal_GPU) {
	////////////////////////////////////////
		

		//allocate filter memory in GPU and init its value 			
		T **host_2d_filter = new T *[filter_num];		
		for (int i = 0; i < filter_num; i++) {
			T *host_1d=new T [filter_size*filter_size*channel_num];
			host_1d = filter_array[i];
			T *dev_1d;			
			cudaMalloc((void **)&dev_1d, sizeof(T)*filter_size*filter_size*channel_num);
			cudaMemcpy(dev_1d,host_1d, sizeof(T)*filter_size*filter_size*channel_num, cudaMemcpyHostToDevice);			
			host_2d_filter[i] = dev_1d;
			
			//delete[] host_1d;
			cudaFree(dev_1d);
		}
		cudaMalloc((void **)&cuda_filter, sizeof(T*)*filter_num);
		cudaMemcpy(cuda_filter, host_2d_filter, sizeof(T *)*filter_num, cudaMemcpyHostToDevice);
		delete[] host_2d_filter;

		//allocate input memory in GPU and init its value
		T **host_2d_input = new T *[filter_size*filter_size*channel_num];
		for (int i = 0; i < filter_size*filter_size*channel_num; i++) {
			T *host_1d = new T[input_width*input_hight];
			host_1d = input_array[i];
			T *dev_1d;
			cudaMalloc((void **)&dev_1d, sizeof(T)*input_hight*input_width);
			cudaMemcpy(dev_1d, host_1d, sizeof(T)*input_hight*input_width, cudaMemcpyHostToDevice);
			host_2d_input[i] = dev_1d;

			//delete[] host_1d;
			cudaFree(dev_1d);
		}
		cudaMalloc((void **)&cuda_input, sizeof(T*)*filter_size*filter_size*channel_num);
		cudaMemcpy(cuda_input, host_2d_input, sizeof(T *)*filter_size*filter_size*channel_num, cudaMemcpyHostToDevice);
		delete[] host_2d_input;


		//allocate output result memory in GPU
		T **host_2d_output = new T *[filter_num];
		for (int i = 0; i < filter_num; i++) {			
			//T *hostyyy_1d = new T[input_hight*input_width];
			T *dev_1d;
			cudaMalloc((void **)&dev_1d, sizeof(T)*input_hight*input_width);
			//cudaMemcpy(dev_1d, output_tmp[i], sizeof(T)*input_hight*input_width, cudaMemcpyHostToDevice);
			host_2d_output[i] = dev_1d;

			//delete[] hostyyy_1d,dev_1d;
			cudaFree(dev_1d);
		}
		cudaMalloc((void **)&cuda_output, sizeof(T*)*filter_num);
		cudaMemcpy(cuda_output, host_2d_output, sizeof(T *)*filter_num, cudaMemcpyHostToDevice);
		delete[] host_2d_output;

		//starting tick here
		
		//float elapsedTime;
		//cudaEventCreate(&start);
		//cudaEventRecord(start, 0);

		
		
		//cudaEventSynchronize(stop);
		//Functionname<<<block number, thread number, share memory size>>>(variable…)
		cudaEventRecord(start, 0);		
		dim3 grid(filter_num, 1, 1), block(filter_size*filter_size*channel_num, 1, 1);	
		Mult_normal <<< grid, block >>> (cuda_filter, cuda_input, cuda_output, filter_num, filter_size, channel_num, input_hight, input_width);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);		
		float time_elapsed = 0;
		cudaEventElapsedTime(&time_elapsed, start, stop);		
		GPU_time.push_back(time_elapsed * 1000);

		//copy result from GMEM to MEM
		//host_2d_output = new T *[filter_num];
		//cudaMemcpy(host_2d_output, cuda_output, sizeof(T *)*filter_num, cudaMemcpyDeviceToHost);
		
		cudaMemcpy(output_tmp, cuda_output, sizeof(T)*filter_num*input_width*input_hight, cudaMemcpyDeviceToHost);

		

		cout << "*****************************" << endl;
		cout << "The following is GPU result:" << endl;
		for (int i = 0; i < filter_num; i++) {
			for (int j = 0; j < input_width*input_hight; j++) {
				cout << output_tmp[i][j] << " ";
			}
			cout << endl;
		}
		
		//release cuda mem on
		//cuda_filter

		cudaFree(cuda_filter);
		cudaFree(cuda_input);
		cudaFree(cuda_output);

		
		
		

	//////////////////////////////////////////	
	}
	else {
	/////////////////////////////////////////
	exit(0);
	/////////////////////////////////////////
	}
}



template<class T>
void con_mul<T>::output_transform() {
	cout << "The following is reshaped results" << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int j = 0; j < input_hight; j++) {
			for (int m = 0; m < input_width; m++) {
				output[i][j][m] = output_tmp[i][j*input_width + m];
				cout << output[i][j][m] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	
}


template<class T>
void con_mul<T>::deal_with_time() {
	cout << "CPU time:" << endl;
	for (int i = 0; i < CPU_time.size(); i++)
		printf("%f us\n", CPU_time[i]);
		//cout << CPU_time[i]<<"ms" << endl;

	cout <<endl<< "GPU time:" << endl;
	for (int i = 0; i < GPU_time.size(); i++)
		printf("%f us\n", GPU_time[i]);
		//cout << GPU_time[i]<<"ms" << endl;
}


int main() {
	cout << "good afternoon" << endl;
	con_mul<int> pray_no_bug(3, 512, 1, 2, 2, two_four, normal_GPU);
	//filter size, channel number, filter number, input hight
	// input width, matrix sparsity, model

	//preparing data
	pray_no_bug.gpu_info();
	pray_no_bug.matrix_generation();
	pray_no_bug.matrix_transform();

	//calculation
	for(int i=0;i<10;i++)
		pray_no_bug.kernel_CPU();
	pray_no_bug.output_transform();

	//for(int i=0;i<10;i++)
	
	
	
	for(int i=0;i<10;i++)
		pray_no_bug.kernel_GPU();
	
	

	pray_no_bug.output_transform();
	//prepare output

	pray_no_bug.deal_with_time();

	return 0;

}