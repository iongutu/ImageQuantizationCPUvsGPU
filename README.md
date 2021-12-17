# Image Quantization CPU vs GPU

This project was developed in ordered to compare the differences in time of execution between CPU and GPU for Image Quantization using K-Means clustering.


The CPU implementation was writen in C++, and GPU was developed using CUDA platform.

The centroids was selected as an average of intensity for a pixel selected from a uniform distribution in image.

The method of accesing pixels of an image in GPU is maded using two threads from different blocks, and image is sent as an array (see image below)

![image](https://user-images.githubusercontent.com/62872057/146520912-439a49f0-61c4-4d7e-aa9c-0ff09f7f2293.png)

HW-specification
  GPU: Nvidia 1660 TI
  CPU: Intel i7-9750H

After run the Algorithm with different data input and with different nr. of clusters, GPU's performance is undeniable.
![image](https://user-images.githubusercontent.com/62872057/146521513-5f4f5f2b-a09f-4e09-b211-2ce1aa67d80d.png)

The mean time for analyze each pixel in an image for GPU is considerably smaller than for CPU.
![image](https://user-images.githubusercontent.com/62872057/146522469-707addce-7da5-44f5-82b8-93f94a7e1084.png)

##As image quantization preserves only the most  K important colors in an image,it can be used as a basic segmentation method.
![image](https://user-images.githubusercontent.com/62872057/146523916-cc05974f-685e-4af7-a2d4-ebdd7de0f65d.png)


