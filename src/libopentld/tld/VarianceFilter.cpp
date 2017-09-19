/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
* VarianceFilter.cpp
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#include "VarianceFilter.h"

#include "IntegralImage.h"
#include "DetectorCascade.h"
#include "TLDUtil.h"
#include <iostream>
#include <fstream>
#include "switch.h"

using namespace std;



using namespace cv;
using namespace std;

namespace tld
{

	VarianceFilter::VarianceFilter()
	{
		enabled = true;
		minVar = 0;
		integralImg = NULL;
		integralImg_squared = NULL;

	}

	VarianceFilter::~VarianceFilter()
	{
		release();
	}

	void VarianceFilter::release()
	{
		if (integralImg != NULL) delete integralImg;

		integralImg = NULL;

		if (integralImg_squared != NULL) delete integralImg_squared;

		integralImg_squared = NULL;

	}

	int VarianceFilter::convertToString(const char *filename, std::string& s)
	{
		size_t size;
		char*  str;
		std::fstream f(filename, (std::fstream::in | std::fstream::binary));

		if (f.is_open())
		{
			size_t fileSize;
			f.seekg(0, std::fstream::end);
			size = fileSize = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);
			str = new char[size + 1];
			if (!str)
			{
				f.close();
				return 0;
			}

			f.read(str, fileSize);
			f.close();
			str[size] = '\0';
			s = str;
			delete[] str;
			return 0;
		}
		cout << "Error: failed to open file\n:" << filename << endl;
		return false;
	}

	float VarianceFilter::calcVariance(int *off)
	{

		int *ii1 = integralImg->data;
		float  *ii2 = integralImg_squared->data;

		float mX = (ii1[off[3]] - ii1[off[2]] - ii1[off[1]] + ii1[off[0]]) / (float)off[5]; //Sum of Area divided by area
		float mX2 = (ii2[off[3]] - ii2[off[2]] - ii2[off[1]] + ii2[off[0]]) / (float)off[5];
		return mX2 - mX * mX;
	}

	void VarianceFilter::integralImag_opencv(const Mat &img)
	{



		///DevicesInfo devices;
		//getOpenCLDevices(devices);
		//setDevice(devices[SELECTED_DEVICE_ID]);
		oclMat img1, Sum, sqsum;

		img1 = img;


		cv::ocl::integral(img1, Sum, sqsum);
		iSumMat = Sum;
		fSqreSumMat = sqsum;

		//int *p2, *q2;
		//float *fp2, *fq2;
		//p2 = (int*) iSumMat.data;
		//fp2 = (float*)fSqreSumMat.data;

		//for (int i = 0; i < 1; i++)
		//{
		//	for (int m = 0; m < 10; m++)
		//	{
		//		q2 = p2 + i*img.cols + m;
		//		fq2 = fp2 + i*img.cols + m;
		//		//	printf("%f  ", (float)(*q2));

		//	}
		//	printf("\n");
		//}



		//double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
		//toc = toc / 1000000;
		//printf("            the copy data  used is %f************\n", toc);//float fps = 1 / toc;
		//																   //  const unsigned char *input = (const unsigned char *)(img.data);



	}

	void VarianceFilter::integralImag_extract(const Mat &src, Mat &sum, Mat &sqsum)//void VarianceFilter::integralImag_extract(const Mat &src)
	{


		int i, j;

		cv::Mat t_sum;
		//cv::Mat sum;
		cv::Mat t_sqsum;
		//cv::Mat sqsum;

		int vlen = 4; cl_event events[2];
		int offset = 0;
		int pre_invalid = 0;
		int vcols = (pre_invalid + src.cols + vlen - 1) / vlen;

		int w = src.cols, h = src.rows;
		int depth = src.depth() == CV_8U ? CV_32S : CV_64F;
		int type = CV_MAKE_TYPE(depth, 1);

#ifdef debug
		printf("the type is %d\n", depth);
#endif
		t_sum.create(src.cols, src.rows, type);
		sum.create(h, w, type);

		t_sqsum.create(src.cols, src.rows, CV_32FC1);
		sqsum.create(h, w, CV_32FC1);

		int sum_offset = 0;    // sum.offset / vlen;
		int sqsum_offset = 0;  // sqsum.offset / vlen;


		cl_mem srcdata = clCreateBuffer(context, CL_MEM_READ_ONLY, (src.cols) * (src.rows) * sizeof(uchar), NULL, NULL);
		cl_mem tsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (src.cols) * (src.rows) * sizeof(int), NULL, NULL);
		cl_mem tsqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (src.cols) * (src.rows) * sizeof(float), NULL, NULL);
		cl_mem sumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (src.cols) * (src.rows) * sizeof(int), NULL, NULL);
		cl_mem sqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (src.cols) * (src.rows) * sizeof(float), NULL, NULL);
		//use host ptr
		//cl_mem srcdata = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (src.cols) * (src.rows) * sizeof(uchar), (void *)src.data, NULL);
		//cl_mem tsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (src.cols) * (src.rows) * sizeof(int), (void *)t_sum.data, NULL);
		//cl_mem tsqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE  | CL_MEM_USE_HOST_PTR , (src.cols) * (src.rows) * sizeof(float), (void *)t_sqsum.data, NULL);
		//cl_mem sumdata = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (src.cols) * (src.rows) * sizeof(int), (void *)sum.data, NULL);
		//cl_mem sqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (src.cols) * (src.rows) * sizeof(float), (void *)sqsum.data, NULL);

		int step;
		int src_step, sum_step;
		int t_sum_step, sqsum_step;

		//src_step = 1920;
		//t_sum_step =  4320;
		//sum_step = 7680;
		//sqsum_step =  7680;
		int rows = src.rows;
		int cols = src.cols;
		src_step = src.step;
		t_sum_step = t_sum.step;
		sum_step = sum.step;
		sqsum_step = sqsum.step;

		//printf("");
		status = clEnqueueWriteBuffer(commandQueue, srcdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(uchar), src.data, 0, NULL, NULL);

		assert(status == CL_SUCCESS);
		//if (status != CL_SUCCESS)
		//{
		//	cout << "Error: src.data kernel_intgegral_cols EnqueueNDRangeKernel!" << endl;
		//}

		status = clSetKernelArg(kernel_intgegral_cols, 0, sizeof(cl_mem), (void *)&srcdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&src.data));
		status = clSetKernelArg(kernel_intgegral_cols, 1, sizeof(cl_mem), (void *)&tsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sum.data));
		status = clSetKernelArg(kernel_intgegral_cols, 2, sizeof(cl_mem), (void *)&tsqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sqsum.data));
		status = clSetKernelArg(kernel_intgegral_cols, 3, sizeof(cl_int), (void *)&offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&offset));
		status = clSetKernelArg(kernel_intgegral_cols, 4, sizeof(cl_int), (void *)&pre_invalid);//args.push_back(make_pair(sizeof(cl_int), (void *)&pre_invalid));
		status = clSetKernelArg(kernel_intgegral_cols, 5, sizeof(cl_int), (void *)&rows);//args.push_back(make_pair(sizeof(cl_int), (void *)&src.rows));
		status = clSetKernelArg(kernel_intgegral_cols, 6, sizeof(cl_int), (void *)&cols);//args.push_back(make_pair(sizeof(cl_int), (void *)&src.cols));
		status = clSetKernelArg(kernel_intgegral_cols, 7, sizeof(cl_int), &src_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&src.step));
		status = clSetKernelArg(kernel_intgegral_cols, 8, sizeof(cl_int), &t_sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.step));



		size_t gt[3] = { ((vcols + 1) / 2) * 256, 1, 1 }, lt[3] = { 256, 1, 1 };
		//	//openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_cols", gt, lt, args, -1, depth);
		status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_cols, 1, NULL, gt, lt, 0, NULL, &events[0]);
		status = clWaitForEvents(1, &events[0]);
		assert(status == CL_SUCCESS);
		//	
		//	status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_cols, 1, NULL, gt, lt, 0, NULL, NULL);
		//
		//if (status != CL_SUCCESS)
		//{
		//	cout << "Error: kernel_intgegral_cols EnqueueNDRangeKernel!" << endl;
		//}
		//	status = clWaitForEvents(1, &events[0]);

		//if (status != CL_SUCCESS)
		//	{
		//	printf("Error: Waiting for kernel run to finish.	(clWaitForEvents0)\n");
		//	}

		/*status = clEnqueueReadBuffer(commandQueue, sumdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(int),  t_sum.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clEnqueueReadBuffer(commandQueue, sqsumdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(int),  t_sqsum.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clEnqueueWriteBuffer(commandQueue, sumdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(int),  t_sum.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clEnqueueWriteBuffer(commandQueue, sqsumdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(int),  t_sqsum.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		*/

		//clReleaseMemObject(tsumdata);
		//clReleaseMemObject(tsqsumdata);


		// tsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (src.cols) * (src.rows) * sizeof(int), (void *)t_sum.data, NULL);
		//  tsqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (src.cols) * (src.rows) * sizeof(float), (void *)t_sqsum.data, NULL);



		status = clSetKernelArg(kernel_intgegral_rows, 0, sizeof(cl_mem), (void *)&tsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sum.data));
		status = clSetKernelArg(kernel_intgegral_rows, 1, sizeof(cl_mem), (void *)&tsqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sqsum.data));
		status = clSetKernelArg(kernel_intgegral_rows, 2, sizeof(cl_mem), (void *)&sumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&sum.data));
		status = clSetKernelArg(kernel_intgegral_rows, 3, sizeof(cl_mem), (void *)&sqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&sqsum.data));
		status = clSetKernelArg(kernel_intgegral_rows, 4, sizeof(cl_int), (void *)&t_sum.rows);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.rows));
		status = clSetKernelArg(kernel_intgegral_rows, 5, sizeof(cl_int), (void *)&t_sum.cols);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.cols));
		status = clSetKernelArg(kernel_intgegral_rows, 6, sizeof(cl_int), (void *)&t_sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.step));
		status = clSetKernelArg(kernel_intgegral_rows, 7, sizeof(cl_int), (void *)&sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&sum.step));
		status = clSetKernelArg(kernel_intgegral_rows, 8, sizeof(cl_int), (void *)&sqsum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&sqsum.step));
		status = clSetKernelArg(kernel_intgegral_rows, 9, sizeof(cl_int), (void *)&sum_offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&sum_offset));
		status = clSetKernelArg(kernel_intgegral_rows, 10, sizeof(cl_int), (void *)&sqsum_offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&sqsum_offset));

		size_t gt2[3] = { t_sum.cols * 32, 1, 1 }, lt2[3] = { 256, 1, 1 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_rows, 1, NULL, gt2, lt2, 0, NULL, &events[1]);//openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_rows", gt2, lt2, args, -1, depth);


																													 //if (status != CL_SUCCESS)
																													 //{
																													 //	cout << "Error: kernel_intgegral_rows EnqueueNDRangeKernel!" << endl;
																													 //}
		status = clWaitForEvents(1, &events[1]);
		assert(status == CL_SUCCESS);
		//if (status != CL_SUCCESS)
		//{
		//	printf("Error: Waiting for kernel run to finish.	(clWaitForEvents0)\n");
		//}

		//convert clMat to mat
		status = clEnqueueReadBuffer(commandQueue, sumdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(int), sum.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clEnqueueReadBuffer(commandQueue, sqsumdata, CL_FALSE, 0, (src.cols) * (src.rows) * sizeof(int), sqsum.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);


		//psum = (float*) &sqsum.data;
		//for (i = 0; i <src.cols; i++)
		//{
		//	for (j = 0; j < src.rows; j++)
		//	{
		//		 cout <<psum[src.cols*j+i]<<endl;
		//	}
		//}
		//for (j = 0; j <10; j++)
		// 	{
		// 		 cout <<psum[j]<<endl;
		// 	}

		//iSumMat = sum;
		//fSqreSumMat = sqsum;
		//iSumMat.data[0] = 100;
		t_sum.release();
		t_sqsum.release();
		//sum.release();
		//sqsum.release();
		//	clReleaseKernel(kernel_intgegral_rows);
		//	clReleaseKernel(kernel_intgegral_cols);
		//	clReleaseProgram(program);
		clReleaseMemObject(srcdata);
		clReleaseMemObject(tsumdata);
		clReleaseMemObject(tsqsumdata);
		clReleaseMemObject(sumdata);
		clReleaseMemObject(sqsumdata);
		clReleaseEvent(events[0]);
		clReleaseEvent(events[1]);


		//status = clReleaseCommandQueue(commandQueue);	 
		//status = clReleaseContext(context);				 





	}




	void VarianceFilter::nextIteration(const Mat &img)
	{
		if (!enabled) return;

		release();
#if CPU
		  integralImg = new IntegralImage<int>(img.size());
		 integralImg_squared = new IntegralImage<float>(img.size());
#endif
#if PrintTime_integral
		double tic = cvGetTickCount();
#endif
//#if    clIntegral_m_1
//		//integralImg->oclcalcIntImg(img);
//		//integralImg_squared->oclcalcIntImgSquare(img, true);
//
//		oclMat temp;
//		temp = img;
//		jifentu2(temp, sum, sqsum);   //(m+1,n+1)-->(m,n)

#if clIntegral_m
		//integralImag_opencv( img);    // (m,n)-->(m,n)
		integralImag_extract(img, iSumMat, fSqreSumMat);
		//integralImag_extract(img);
		//integralImg->calcIntImg(img);
		//integralImg_squared->calcIntImg(img, true);
#elif CPU
		integralImg->calcIntImg(img);
		integralImg_squared->calcIntImg(img, true);
#endif

		//	int size = img.size().height*img.size().width;
		//for (int i = 0; i < size; i++)
		//{
		//#if clIntegral_m
		//		int img_dat = (int)iSumMat.data[i];
		//		float img_dat2 = (float)fSqreSumMat.data[i];
		//		int temp3 = img_dat;
		//#else
		//		int img_dat = (int)integralImg->data[i];
		//		float img_dat2 = (float)integralImg_squared->data[i];
		//		int temp3 = img_dat;
		//
		//#endif
		//	}

#if PrintTime_integral
		double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

		toc = toc / 1000000;
		printf("*****the clIntegral time used is %f************\n", toc);//float fps = 1 / toc;
#endif	 


	}

	bool VarianceFilter::filter(int i)
	{
		if (!enabled) return true;

		//float bboxvar = calcVariance(windowOffsets + TLD_WINDOW_OFFSET_SIZE * i);
		float bboxvar;
		int *off = windowOffsets + TLD_WINDOW_OFFSET_SIZE * i;
		//
		///
		int *ii1 = integralImg->data;
		float  *ii2 = integralImg_squared->data;

		float mX = (ii1[off[3]] - ii1[off[2]] - ii1[off[1]] + ii1[off[0]]) / (float)off[5]; //Sum of Area divided by area
		float mX2 = (ii2[off[3]] - ii2[off[2]] - ii2[off[1]] + ii2[off[0]]) / (float)off[5];
		bboxvar = mX2 - mX * mX;


		detectionResult->variances[i] = bboxvar;
		//test for i
		//if (i == 1172655)
		//	printf("i==%d,bboxvar=%f\n", i, bboxvar);
		if (bboxvar < minVar)
		{
			return false;
		}

		return true;
	}






	bool VarianceFilter::clfilter(const Mat &img)
	{
		//printf("begin using VarianceFilter clFilter*************************\n");

		//char *kernelName = "M:\\OpenTLD\\OpenTLD-master\\OpenTLD-master\\src\\libopentld\\tld\\varianceFilter.cl";
		char *kernelName = "..\\..\\..\\src\\libopentld\\kernel\\varianceFilter_kernel.cpp"; // it is conbined to ensemble classify.

		cl_event events[1];
		string sourceStr;
		status = convertToString(kernelName, sourceStr);
		const char *source = sourceStr.c_str();
		size_t sourceSize[] = { strlen(source) };
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

		/*Step 6: Build program. */
		status = clBuildProgram(program, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);
		//printf("mid using GPU*************************\n");
		if (status != CL_SUCCESS)
		{
			cout << "VarianceFilter Error: Getting platforms!" << endl;
			return false;
		}

		cl_mem oclbuffWindowsOffset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (TLD_WINDOW_OFFSET_SIZE * numWindows) * sizeof(int), (void *)windowOffsets, NULL);
		cl_mem oclbuffII = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (img.size().width)*(img.size().height) * sizeof(int), (void *)iSumMat.data, NULL);
		cl_mem oclbuffIISqure = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (img.size().width)*(img.size().height) * sizeof(int), (void *)fSqreSumMat.data, NULL);
		cl_mem oclbuffDetectionResultVarious = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)detectionResult->variances, NULL);
		cl_mem oclbuffDetectionResultPosteriors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)detectionResult->posteriors, NULL);

		cl_mem oclbuffDetectionwindowFlags = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(int), (void *)detectionResult->windowFlags, NULL);

		/*Step 8: Create kernel object */
		kernel = clCreateKernel(program, "varianceFilter", NULL);

		/*Step 9: Sets Kernel arguments.*/
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&oclbuffWindowsOffset);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&oclbuffII);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&oclbuffIISqure);
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&oclbuffDetectionResultVarious);
		status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oclbuffDetectionResultPosteriors);
		status = clSetKernelArg(kernel, 5, sizeof(int), (void *)&numWindows);
		status = clSetKernelArg(kernel, 6, sizeof(float), (void *)&minVar);
		status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&oclbuffDetectionwindowFlags);
		//status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&TLD_WINDOW_OFFSET_SIZE);

		/*Step 10: Running the kernel.*/
		printf("begore opencl kernel numWindows=%d\n", numWindows);
		size_t global_work_size[1] = { numWindows };
		size_t local_work_size[1] = { 256 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &events[0]);
		if (status != CL_SUCCESS)
		{
			cout << "Error:VarianceFilter EnqueueNDRangeKernel!" << endl;
			//return false;
		}
		status = clWaitForEvents(1, &events[0]);

		if (status != CL_SUCCESS)
		{
			printf("Error: Waiting for kernel run to finish.	(clWaitForEvents0)\n");

		}
		//cout << "o" << endl;

		status = clReleaseEvent(events[0]);

		//printf("end using GPU*************************\n");
		//for (int i = 0; i < numWindows; i++)
		//	if(detectionResult->windowFlags[i]==1)
		//	printf("detectionResult[%d] is %d\n", i,detectionResult->windowFlags[i]);

		/*Step 11: Read the cout put back to host memory.*/
		//status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, 12 * sizeof(char), output, 0, NULL, NULL);



		clReleaseKernel(kernel);
		clReleaseProgram(program);

		clReleaseMemObject(oclbuffWindowsOffset);
		clReleaseMemObject(oclbuffII);
		clReleaseMemObject(oclbuffIISqure);
		clReleaseMemObject(oclbuffDetectionResultVarious);
		clReleaseMemObject(oclbuffDetectionResultPosteriors);
		clReleaseMemObject(oclbuffDetectionwindowFlags);
		return true;


	}
} /* namespace tld */
