#include <climits>
#include <cmath>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <vector_types.h>
#include <vector>
#include <chrono>

#define MAX_CLASSES 96
#define MAX_BANDS 3  // RGB

struct double3x3 {
    double3 m[3];
};

double3 h_means[MAX_CLASSES];
double3x3 h_invCov[MAX_CLASSES];
double h_logDet[MAX_CLASSES];

void classifyMLCCPU(
    uchar4* pixels,
    int N,
    int numClasses
) {
    for(int i = 0; i < N; ++i) {
        double R = pixels[i].x;
        double G = pixels[i].y;
        double B = pixels[i].z;

        double bestVal = -1e30f;
        int bestClass = 0;

        for (int k = 0; k < numClasses; ++k) {
            double dR = R - h_means[k].x;
            double dG = G - h_means[k].y;
            double dB = B - h_means[k].z;

            double quad = dR*(h_invCov[k].m[0].x*dR + h_invCov[k].m[0].y*dG + h_invCov[k].m[0].z*dB)
                    + dG*(h_invCov[k].m[1].x*dR + h_invCov[k].m[1].y*dG + h_invCov[k].m[1].z*dB)
                    + dB*(h_invCov[k].m[2].x*dR + h_invCov[k].m[2].y*dG + h_invCov[k].m[2].z*dB);

            double L = -0.5f * h_logDet[k] - 0.5f * quad;

            if (L > bestVal) {
                bestVal = L;
                bestClass = k;
            }
        }

        pixels[i].w = bestClass;
    }
}

double3 computeMean(const uchar4* pixels, const std::vector<int>& indices) {
    double3 mean = {0.f, 0.f, 0.f};
    int n = indices.size();
    for (int idx : indices) {
        uchar4 p = pixels[idx];
        mean.x += static_cast<double>(p.x);
        mean.y += static_cast<double>(p.y);
        mean.z += static_cast<double>(p.z);
    }
    mean.x /= n;
    mean.y /= n;
    mean.z /= n;
    return mean;
}

double3x3 computeInvCov(const uchar4* pixels, const std::vector<int>& indices, const double3& mean, double* logDet) {
    double cov[3][3] = {{0.f,0.f,0.f},{0.f,0.f,0.f},{0.f,0.f,0.f}};
    int n = indices.size();

    for (int idx : indices) {
        uchar4 p = pixels[idx];
        double dR = static_cast<double>(p.x) - mean.x;
        double dG = static_cast<double>(p.y) - mean.y;
        double dB = static_cast<double>(p.z) - mean.z;

        cov[0][0] += dR*dR; cov[0][1] += dR*dG; cov[0][2] += dR*dB;
        cov[1][0] += dG*dR; cov[1][1] += dG*dG; cov[1][2] += dG*dB;
        cov[2][0] += dB*dR; cov[2][1] += dB*dG; cov[2][2] += dB*dB;
    }

    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            cov[i][j] /= (n-1);

    double a=cov[0][0], b=cov[0][1], c=cov[0][2],
          d=cov[1][0], e=cov[1][1], f=cov[1][2],
          g=cov[2][0], h=cov[2][1], i=cov[2][2];

    double det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    *logDet = std::log(det);

    double3x3 inv;
    inv.m[0].x =  (e*i - f*h)/det; inv.m[0].y = -(b*i - c*h)/det; inv.m[0].z =  (b*f - c*e)/det;
    inv.m[1].x = -(d*i - f*g)/det; inv.m[1].y =  (a*i - c*g)/det; inv.m[1].z = -(a*f - c*d)/det;
    inv.m[2].x =  (d*h - e*g)/det; inv.m[2].y = -(a*h - b*g)/det; inv.m[2].z =  (a*e - b*d)/det;

    return inv;
}

int main() {
    std::string in_file, out_file;
    std::cin >> in_file >> out_file;
    int nc;
    std::cin >> nc;
    std::vector<std::vector<int>> classIndices(nc);

    int w, h;
    FILE *fp = fopen(in_file.c_str(), "rb");
    if (!fp) {
        std::cerr << "No such file" << std::endl;
        return 0;
    }

    fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
    uchar4 *pixels = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(pixels, sizeof(uchar4), w * h, fp);
    fclose(fp);

    for(int k=0;k<nc;k++){
        int npj; std::cin >> npj;
        classIndices[k].resize(npj);
        for(int j=0;j<npj;j++){
            int x,y; std::cin >> x >> y;
            classIndices[k][j] = y*w + x;
        }
    }

    int N = w*h;

    for (int k = 0; k < nc; k++) {
        double3 mean = computeMean(pixels, classIndices[k]);
        h_means[k] = {mean.x, mean.y, mean.z};
        h_invCov[k] = computeInvCov(pixels, classIndices[k], mean, &h_logDet[k]);
    }

    const double num_runs = 20;  // Количество прогонов для усреднения
    float min_avg_time = 10;
    int min_blocks = 0, min_threads = 0;
    int total_time = 0;
    int min_time = INT_MAX;
    int max_time = 0;

    // Выводим заголовок CSV
    printf("threads,blocks,total_threads,avg_time_ms,min_time_ms,max_time_ms\n");

    for (int run = 0; run < num_runs; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        classifyMLCCPU(pixels, N, nc);

        auto stop = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float, std::milli>(stop - start).count(); // ms

        total_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }

    printf("avg_time = %f ms", total_time / num_runs);

    // fp = fopen(out_file.c_str(), "wb");
    // fwrite(&w, sizeof(int), 1, fp);
    // fwrite(&h, sizeof(int), 1, fp);
    // fwrite(result, sizeof(uchar4), N, fp);
    // fclose(fp);

    return 0;
}