#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)


__global__ void kernel(const double* __restrict__ arr1, const double* __restrict__ arr2, double* __restrict__ res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < size) {
        res[idx] = fmax(arr1[idx], arr2[idx]);
        idx += offset;
    }
}

int main() {
    int n;
    std::cin >> n;
    double *arr1 = (double *)malloc(sizeof(double) * n);
    if (!arr1) {
      return 0;
    }
    double *arr2 = (double *)malloc(sizeof(double) * n);
    if (!arr2) {
      return 0;
    }
    double *arr_res = (double *)malloc(sizeof(double) * n);
    if (!arr_res) {
      return 0;
    }
    for (int i = 0; i < n; ++i) {
      std::cin >> arr1[i];
    }
    for (int i = 0; i < n; ++i) {
      std::cin >> arr2[i];
    }
    double *dev_arr1, *dev_arr2, *dev_res;
    CSC(cudaMalloc(&dev_arr1, sizeof(double) * n));
    CSC(cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_arr2, sizeof(double) * n));
    CSC(cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_res, sizeof(double) * n));
  
    const int num_runs = 20;  // Количество прогонов для усреднения
    float min_avg_time = 10;
    int min_blocks = 0, min_threads = 0;
    
    // Выводим заголовок CSV
    printf("threads,blocks,total_threads,avg_time_ms,min_time_ms,max_time_ms\n");
    
    for (int i = 32; i < 1025; i += 32) {
      for (int j = 32; j < 1025; j += 32) {
        float total_time = 0.0f;
        float min_time = 10;
        float max_time = 0.0f;
        
        // Запускаем kernel num_runs раз для усреднения
        for (int run = 0; run < num_runs; run++) {
          cudaEvent_t start, stop;
          CSC(cudaEventCreate(&start));
          CSC(cudaEventCreate(&stop));
          CSC(cudaEventRecord(start));
          kernel<<<j, i>>>(dev_arr1, dev_arr2, dev_res, n);
          CSC(cudaEventRecord(stop));
          CSC(cudaEventSynchronize(stop));
          CSC(cudaGetLastError());
      
          float t;
          CSC(cudaEventElapsedTime(&t, start, stop));
          CSC(cudaEventDestroy(start));
          CSC(cudaEventDestroy(stop));
          
          total_time += t;
          if (t < min_time) min_time = t;
          if (t > max_time) max_time = t;
        }
        
        float avg_time = total_time / num_runs;
        
        if (avg_time < min_avg_time) {
          min_blocks = i;
          min_threads = j;
          min_avg_time = avg_time;
        }
        
        // Выводим в формате CSV
        printf("%d,%d,%d,%.6f,%.6f,%.6f\n", i, j, i * j, avg_time, min_time, max_time);
      }
    }

    printf("MINIMAL avg_time = %f ms, block = %d, threads = %d\n", min_avg_time, min_blocks, min_threads);

    CSC(cudaDeviceSynchronize());
    CSC(cudaMemcpy(arr_res, dev_res, sizeof(double) * n, cudaMemcpyDeviceToHost));
    // for(int i = 0; i < n; i++)
    //     printf("%.10e ", arr_res[i]);
    // printf("\n");

    CSC(cudaFree(dev_arr1));
    CSC(cudaFree(dev_arr2));
    CSC(cudaFree(dev_res));
    free(arr1);
    free(arr2);
    free(arr_res);
    return 0;
}