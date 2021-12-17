#include<iostream>
#include<opencv2\opencv.hpp>
#include <vector>
#include <random>

//for time measuring 
#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace cv;

//value to always generate random uniform distributed numbers;
random_device dev;
mt19937 random(dev());



//euclidian distance in 3 dimensions
double dist3dVectors(int x1, int y1, int z1, int x2, int y2, int z2)
{
	return sqrt(((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2)) + ((z1 - z2)*(z1 - z2)));
}

int randomUniformValue(int minValue, int maxValue)
{

	uniform_int_distribution<mt19937::result_type> distribution(minValue, maxValue);
	return distribution(random);

}


class Pix {
private:
	uchar blue, green, red;

public:
	Pix(uchar, uchar, uchar);
	uchar getBlue();
	uchar getGreen();
	uchar getRed();
};


class Kmenas
{
private:
	vector<Pix> centroids; // coordinate of centroids
	Mat image;
	Mat label;//label of each pixel
	int K; // number of clusters;

public:
	Kmenas(Mat img, int k)
	{
		image = img;
		K = k;
		//set to all pixel unspecific cluster label;
		label = Mat::zeros(image.rows, image.cols, CV_8UC1);
		// Set random position to centroids
		for (int i = 0; i < K; i++)
		{
			int row = randomUniformValue(0, image.rows - 1);
			int col = randomUniformValue(0, image.cols - 1);
			Vec3b centroidPixel = image.at<Vec3b>(row, col);
			centroids.push_back(Pix(uchar(centroidPixel[0]), uchar(centroidPixel[1]), uchar(centroidPixel[2])));
		}
		
	}
	void assignPointToClosestCluster() {
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				int cluster = 0;
				double distance, minDistance=0;
				Vec3b pixel = image.at<Vec3b>(i, j);
				minDistance = dist3dVectors(centroids[0].getBlue(), centroids[0].getGreen(), centroids[0].getRed(), uchar(pixel[0]), uchar(pixel[1]), uchar(pixel[2]));
				for (int k = 1; k < K; k++)
				{
					distance = dist3dVectors(centroids[k].getBlue(), centroids[k].getGreen(), centroids[k].getRed(), uchar(pixel[0]), uchar(pixel[1]), uchar(pixel[2]));
					if (distance < minDistance)
					{
						minDistance = distance;
						// save the cluster label of a pixel in image
						cluster = k;
						label.at<uchar>(i, j) = (uchar)cluster;
					}
				}
			}
		}
	}
	void recalculateCentroids()
	{
		for (int k = 0; k < K; k++)
		{
			double b = 0.0, g = 0.0, r = 0.0;
			long count = 0;
			for (int i = 0; i < image.rows;i++)
			{
				for (int j = 0; j < image.cols; j++)
				{

					if (label.at<uchar>(i, j) == k)
					{
						count++;
						Vec3b pix = image.at<Vec3b>(i, j);
						b += pix[0];
						g += pix[1];
						r += pix[2];
					}

				}
			}
			b /= count;
			g /= count;
			r /= count;

			centroids.at(k) = Pix(b, g, r);
		}
	}



	void getImageBack()
	{
		for (int r = 0; r < image.rows; r++)
		{
			for (int c = 0; c < image.cols; c++)
			{

				Pix p = centroids.at(label.at<uchar>(r, c));
				cv::Vec3b bgr_pixel(p.getBlue(), p.getGreen(), p.getRed());
				image.at<cv::Vec3b>(r, c) = bgr_pixel;
			}
		}
	}

};




Pix::Pix(uchar b, uchar g, uchar r)
{
	blue = b;
	green = g;
	red = r;
}

uchar Pix::getBlue()
{
	return blue;
}

uchar Pix::getGreen()
{
	return green;
}

uchar Pix::getRed()
{
	return red;
}



int main()
{

	int nClusters = 6;
	String outImage = "out.png";
	Mat image = imread("3.jpg");

	if (image.empty())
		return -1;

	Kmenas asa(image, nClusters);

	
	auto start = high_resolution_clock::now();
	asa.assignPointToClosestCluster();
	for (int i = 0;i < 5;++i)
	{
		asa.recalculateCentroids();
		asa.assignPointToClosestCluster();
		//cout << "Train step " << i << " done" << endl;
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time: " << duration.count() << " microseconds" << endl;


	asa.getImageBack();
	imwrite(outImage, image);
	system("sleep");
	getchar();
	return 0;
}