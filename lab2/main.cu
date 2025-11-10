#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <cstring>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)
// #define CSC(call)

__device__ uchar4 calc_average(cudaTextureObject_t tex, int block_w, int block_h, int w, int h, int x, int y) {
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < block_h; ++i) {
        for (int j = 0; j < block_w; ++j) {
            float u = ( (x * block_w + j) + 0.5f ) / float(w);
            float v = ( (y * block_h + i) + 0.5f ) / float(h);

            uchar4 p = tex2D<uchar4>(tex, u, v);

            sum.x += float(p.x);
            sum.y += float(p.y);
            sum.z += float(p.z);
            sum.w += float(p.w);
        }
    }
    const int n = block_w * block_h;
    uchar4 res;
    res.x = uint8_t(sum.x / float(n) + 0.5f);
    res.y = uint8_t(sum.y / float(n) + 0.5f);
    res.z = uint8_t(sum.z / float(n) + 0.5f);
    res.w = uint8_t(sum.w / float(n) + 0.5f);

    return res;
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out,
                       int w, int h,
                       int block_w, int block_h,
                       int new_w, int new_h)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    const int n = block_w * block_h;

    for (int y = idy; y < new_h; y += offset_y) {
        for (int x = idx; x < new_w; x += offset_x) {
            uchar4 avg = calc_average(tex, block_w, block_h, w, h, x, y);
            out[y * new_w + x] = avg;
        }
    }
}

int main() {
    std::string in_file = "in.bin", out_file = "out.bin";
    // std::cin >> in_file >> out_file;
    int w_new = 400, h_new = 400;
    // std::cin >> w_new >> h_new;

    int w, h;
   	FILE *fp = fopen(in_file.c_str(), "rb");
    if (!fp) {
        std::cerr << "No such file" << std::endl;
        return 0;
    }

 	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w_new * h_new));

    printf("threads,blocks,total_threads,avg_time_ms,min_time_ms,max_time_ms\n");

    const int num_runs = 20;  // Количество прогонов для усреднения
    float min_avg_time = 10;
    int min_blocks = 0, min_threads = 0;

    for (int i = 4; i <= 32; i += 4) {
      for (int j = 4; j <= 32; j += 4) {
        // if (j > 32) continue;
        // if (i > 32) continue;
        float total_time = 0.0f;
        float min_time = 10;
        float max_time = 0.0f;

        // Запускаем kernel num_runs раз для усреднения
        for (int run = 0; run < num_runs; run++) {
            cudaEvent_t start, stop;
            CSC(cudaEventCreate(&start));
            CSC(cudaEventCreate(&stop));
            CSC(cudaEventRecord(start));
            kernel<<< dim3(i, i), dim3(j, j) >>>(tex, dev_out, w, h, w / w_new, h / h_new, w_new, h_new);
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

    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w_new * h_new, cudaMemcpyDeviceToHost));

	CSC(cudaDestroyTextureObject(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

    fp = fopen(out_file.c_str(), "wb");
	fwrite(&w_new, sizeof(int), 1, fp);
	fwrite(&h_new, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w_new * h_new, fp);
	fclose(fp);

    free(data);
    return 0;
}