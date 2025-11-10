#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>

struct uchar4 {
    unsigned char x, y, z, w;
    uchar4() : x(0), y(0), z(0), w(0) {}
    uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
        : x(x), y(y), z(z), w(w) {}
};

// Функция для вычисления среднего значения блока пикселей
uchar4 calc_average_cpu(const uchar4* data, int w, int h, int block_w, int block_h, int x, int y) {
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f, sum_w = 0.0f;
    
    for (int i = 0; i < block_h; ++i) {
        for (int j = 0; j < block_w; ++j) {
            int src_x = x * block_w + j;
            int src_y = y * block_h + i;
            
            // Проверяем границы
            if (src_x >= 0 && src_x < w && src_y >= 0 && src_y < h) {
                int idx = src_y * w + src_x;
                sum_x += float(data[idx].x);
                sum_y += float(data[idx].y);
                sum_z += float(data[idx].z);
                sum_w += float(data[idx].w);
            }
        }
    }
    
    const int n = block_w * block_h;
    uchar4 res;
    res.x = (unsigned char)(sum_x / float(n) + 0.5f);
    res.y = (unsigned char)(sum_y / float(n) + 0.5f);
    res.z = (unsigned char)(sum_z / float(n) + 0.5f);
    res.w = (unsigned char)(sum_w / float(n) + 0.5f);
    
    return res;
}

// CPU версия kernel
void kernel_cpu(const uchar4* data, uchar4* out, int w, int h, 
                int block_w, int block_h, int new_w, int new_h) {
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            uchar4 avg = calc_average_cpu(data, w, h, block_w, block_h, x, y);
            out[y * new_w + x] = avg;
        }
    }
}

// CPU версия с многопоточностью (OpenMP)
void kernel_cpu_parallel(const uchar4* data, uchar4* out, int w, int h, 
                         int block_w, int block_h, int new_w, int new_h) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            uchar4 avg = calc_average_cpu(data, w, h, block_w, block_h, x, y);
            out[y * new_w + x] = avg;
        }
    }
}

int main() {
    std::string in_file = "in.bin", out_file = "out_cpu.bin";
    int w_new = 400, h_new = 400;
    
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
    
    std::cout << "Input image: " << w << "x" << h << std::endl;
    std::cout << "Output image: " << w_new << "x" << h_new << std::endl;
    
    uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w_new * h_new);
    
    // Вычисляем размеры блоков
    int block_w = w / w_new;
    int block_h = h / h_new;
    
    std::cout << "Block size: " << block_w << "x" << block_h << std::endl;
    
    // Тестируем однопоточную версию
    std::cout << "\n=== Single-threaded CPU version ===" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    kernel_cpu(data, out, w, h, block_w, block_h, w_new, h_new);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Single-threaded time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Single-threaded time: " << duration.count() / 1000.0 << " milliseconds" << std::endl;
    
    // Тестируем многопоточную версию
    std::cout << "\n=== Multi-threaded CPU version (OpenMP) ===" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    kernel_cpu_parallel(data, out, w, h, block_w, block_h, w_new, h_new);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Multi-threaded time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Multi-threaded time: " << duration.count() / 1000.0 << " milliseconds" << std::endl;
    
    // Сохраняем результат
    fp = fopen(out_file.c_str(), "wb");
    fwrite(&w_new, sizeof(int), 1, fp);
    fwrite(&h_new, sizeof(int), 1, fp);
    fwrite(out, sizeof(uchar4), w_new * h_new, fp);
    fclose(fp);
    
    std::cout << "\nResult saved to " << out_file << std::endl;
    
    free(data);
    free(out);
    return 0;
}
