#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <random>

#define BLOCK_SIZE  32
#define K 10

using namespace cv;
using namespace std;


 //value to always generate random uniform distributed numbers;
random_device dev;
mt19937 random(dev());

int randomUniformValue(int minValue, int maxValue)
{

	uniform_int_distribution<mt19937::result_type> distribution(minValue, maxValue);
	return distribution(random);

}

 // euclidian distance in 3 dimensions
__device__ float dist3dVectors(int x1, int y1, int z1, int x2, int y2, int z2)
{
	return sqrtf(((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2)) + ((z1 - z2)*(z1 - z2)));
}


 // width = column
 __global__ void assignPointToClosestCluster(uchar  * data, uchar* label, uchar * centroids, int width, int height)
 {
	 const int col = blockIdx.x* blockDim.x + threadIdx.x;
	 const int row = blockIdx.y * blockDim.y + threadIdx.y;
	 if (row >= height || col >= width)
		 return;

	 float min_dist, distance;
	 min_dist = dist3dVectors(data[row* width + col], data[row*width + width*height + col], data[row*width + 2 * width*height + col], centroids[0], centroids[1], centroids[2]);
	 int centroid = 0;
	 for (int i = 0; i < K; ++i)
	 {
		 distance = dist3dVectors(data[row* width + col],data[row*width + width*height + col ], data[row*width + 2 *width*height + col] ,centroids[3*i], centroids[3*i + 1], centroids[3*i + 2]);
		 if (distance < min_dist) 
		 {
			 min_dist = distance;
			 centroid = i;
			 label[row* width + col] = centroid;
		 }
	 }

	// label[row* width + col] = centroid;

 }

 __global__ void centroidUpdate(uchar  * data, uchar* label, int width, int height, uint *count, float *newCentroids)
 {
	 const int col = blockIdx.x* blockDim.x + threadIdx.x;
	 const int row = blockIdx.y * blockDim.y + threadIdx.y;
	 if (row >= height || col >= width)
		 return;

	 for (int k = 0; k < K; ++k)
	 {
		 if (label[row* width + col] == k)
		 {

			 atomicAdd(&newCentroids[k * 3], data[row* width + col]);//b
			 atomicAdd(&newCentroids[k * 3 + 1], data[row* width + width*height + col]);//g
			 atomicAdd(&newCentroids[k * 3 + 2], data[row* width  + 2*width*height + col]);//r
			 atomicAdd(&count[k], 1);
		 }
	 }
 }
 

int main()
{

	String outImage = "out.png";
	Mat image = imread("2.jpg");
	Mat chans[3];
	// get each channel of vector separete
	split(image, chans);

	if (image.empty())
		return -1;

	uchar * generalColor  = new uchar[image.rows * image.cols * 3];
	uchar *h_label = new uchar[image.rows * image.cols];

	// Store each channel value in alocated matrix*/
	for (int r = 0; r < image.rows; ++r)
	{
		for (int c = 0; c < image.cols; ++c)
		{
			generalColor[r*image.cols + c] = chans[0].at<uchar>(r,c);
			generalColor[r*image.cols + image.cols*image.rows + c] = chans[1].at<uchar>(r, c);
			generalColor[r*image.cols + 2 * image.cols * image.rows + c] = chans[2].at<uchar>(r, c);
			h_label[r*image.cols +c] = 0;
		}
	}

	/* Alocate space for centroids 
	 * Each centroid will store the intensity of RGB spectr */
	uchar *h_centroids =  new uchar[K * 3];
	uint* h_count = new uint[K];
	//each centroid will have independent spectrum location
	float* h_colors = new float[K*3];

	for (int i = 0; i < K; i++)
	{
		h_count[i] = 0;

		int row = randomUniformValue(0, image.rows - 1);
		int col = randomUniformValue(0, image.cols - 1);
		Vec3b centroidPixel = image.at<Vec3b>(row, col);
		
		for (int j = 0; j < 3;++j) {
			h_centroids[i * 3 + j] = centroidPixel[j];
			h_colors[i * 3 + j] = 0;
		}
	}

	// Alocate resourse for CUDA kernels
	uchar * d_general = 0;
	uchar *d_label = 0;
	uchar *d_centroids = 0;
	uint *d_count = 0;
	float *d_colors = 0;

	cudaMalloc((void **)&d_general, image.rows*image.cols *3* sizeof(uchar));
	cudaMalloc((void **)&d_label, image.rows*image.cols * sizeof(uchar));
	cudaMalloc((void **)&d_centroids, K * 3 * sizeof(uchar));
	cudaMalloc((void **)&d_count, K * sizeof(uint));
	cudaMalloc((void **)&d_colors, K *3 * sizeof(float));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	cudaMemcpy(d_centroids, h_centroids, K * 3* sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_general, generalColor, image.rows*image.cols *3* sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_label, h_label, image.rows*image.cols * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_count, h_count, K * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colors, h_colors, K *3* sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time spent: %f\n", time);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);
	//stop condition
	for (int j = 0; j <10;++j)
	{

		assignPointToClosestCluster << <grid, block >> > (d_general, d_label, d_centroids, image.cols, image.rows);
		cudaDeviceSynchronize();
		centroidUpdate << <grid, block >> > (d_general, d_label, image.cols, image.rows, d_count, d_colors);
		cudaDeviceSynchronize();
		cudaMemcpy(h_count, d_count, K * sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_colors, d_colors, K *3 *sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0;i < K; ++i)
		{
			for(int j = 0; j<3;++j)
				h_centroids[i*3 + j] = h_colors[i*3 + j]/h_count[i];
		}
		cudaMemcpy(d_centroids, h_centroids, K * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
		if (j == 9) {
			cudaMemcpy(h_label, d_label, image.rows*image.cols * sizeof(uchar), cudaMemcpyDeviceToHost);
			
		}	
		cudaMemset(d_count, 0, K * sizeof(uint));
		cudaMemset(d_colors,0, K * 3* sizeof(float));
	}
	
	
	// Compose image
	uchar labelValue;
		for (int r = 0; r < image.rows; r++)
		{
			for (int c = 0; c < image.cols; c++)
			{
				labelValue = h_label[image.cols*r + c];
				uchar blue = h_centroids[labelValue*3];
				uchar green = h_centroids[labelValue*3 + 1];
				uchar red = h_centroids[labelValue*3 + 2];
				cv::Vec3b bgr_pixel(blue, green, red);
				image.at<cv::Vec3b>(r, c) = bgr_pixel;
			}
		}

	imwrite(outImage, image);

	// memory deallocation
	cudaFree(d_centroids);
	cudaFree(d_general);
	cudaFree(d_count);
	cudaFree(d_colors);

	free(generalColor);
	free(h_centroids);
	free(h_count);
	free(h_colors);

	
	getchar();
    return 0;
}