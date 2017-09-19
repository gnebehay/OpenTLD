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
* EnsembleClassifier.cpp
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#include "DetectorCascade.h"
#include <windows.h>

#include <cstdlib>
#include <cmath>

#include <opencv/cv.h>

#include "EnsembleClassifier.h"
#include <conio.h> 
#include <switch.h>

using namespace std;
using namespace cv;
#include <iostream>
#include <fstream>

#define ROUND_K(x,k) ((x+(k-1))/(k))*(k)

namespace tld
{

	//TODO: Convert this to a function
#define sub2idx(x,y,widthstep) ((int) (floor((x)+0.5) + floor((y)+0.5)*(widthstep)))

	EnsembleClassifier::EnsembleClassifier() :
		features(NULL),
		featureOffsets(NULL),
		posteriors(NULL),
		positives(NULL),
		negatives(NULL)
	{
		numTrees = 10;
		numFeatures = 13;
		enabled = true;

	}

	EnsembleClassifier::~EnsembleClassifier()
	{
		release();
	}


	void EnsembleClassifier::init()
	{
		numIndices = pow(2.0f, numFeatures);

		initFeatureLocations();
		initFeatureOffsets();
		initPosteriors();
		int tld_window_offset_size = TLD_WINDOW_OFFSET_SIZE;


		//oclbuffWindowsOffset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (TLD_WINDOW_OFFSET_SIZE * numWindows) * sizeof(int), (void *)windowOffsets, NULL);
		//oclbufffeatureOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (numScales * numTrees * numFeatures * 2) * sizeof(int), (void *)featureOffsets, NULL);
		//oclbuffDetectionResultfeatureVectors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows * numTrees)* sizeof(int), (void *)detectionResult->featureVectors, NULL);
		//oclbuffDetectionResultPosteriors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)detectionResult->posteriors, NULL);
		//
		//oclbuffPosteriors = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numTrees * numIndices)* sizeof(float), (void *)posteriors, NULL);
		//oclbuffDetectionResultVarious = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)detectionResult->variances, NULL);







	}

	void EnsembleClassifier::release()
	{
		delete[] features;
		features = NULL;
		delete[] featureOffsets;
		featureOffsets = NULL;
		delete[] posteriors;
		posteriors = NULL;
		delete[] positives;
		positives = NULL;
		delete[] negatives;
		negatives = NULL;



		clReleaseKernel(variance_ensemble_kernel);
		clReleaseProgram(program);
		clReleaseMemObject(oclbuffWindowsOffset);
		clReleaseMemObject(oclbufffeatureOffsets);
		clReleaseMemObject(oclbuffDetectionResultfeatureVectors);
		clReleaseMemObject(oclbuffDetectionResultPosteriors);
		clReleaseMemObject(oclbuffPosteriors);
		clReleaseMemObject(oclbuffDetectionResultVarious);


	}

	/*
	* Generates random measurements in the format <x1,y1,x2,y2>
	*/
	void EnsembleClassifier::initFeatureLocations()
	{
		int size = 2 * 2 * numFeatures * numTrees;

		features = new float[size];

		for (int i = 0; i < size; i++)
		{
			features[i] = rand() / (float)RAND_MAX;
		}

	}

	//Creates offsets that can be added to bounding boxes
	//offsets are contained in the form delta11, delta12,... (combined index of dw and dh)
	//Order: scale.tree->feature
	void EnsembleClassifier::initFeatureOffsets()
	{

		featureOffsets = new int[numScales * numTrees * numFeatures * 2];
		int *off = featureOffsets;
		long int offsum1, offsum2;

		for (int k = 0; k < numScales; k++)
		{
			Size scale = scales[k];

			for (int i = 0; i < numTrees; i++)
			{
				for (int j = 0; j < numFeatures; j++)
				{

					float *currentFeature = features + (4 * numFeatures) * i + 4 * j;
					offsum1 = sub2idx((scale.width - 1) * currentFeature[0] + 1, (scale.height - 1) * currentFeature[1] + 1, imgWidthStep); //We add +1 because the index of the bounding box points to x-1, y-1
					*off++ = offsum1;

					offsum2 = sub2idx((scale.width - 1) * currentFeature[2] + 1, (scale.height - 1) * currentFeature[3] + 1, imgWidthStep);
					*off++ = offsum2;
				}
			}
		}
	}


	int EnsembleClassifier::convertToString(const char *filename, std::string& s)
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
	void EnsembleClassifier::initPosteriors()
	{
		posteriors = new float[numTrees * numIndices];
		positives = new int[numTrees * numIndices];
		negatives = new int[numTrees * numIndices];

		for (int i = 0; i < numTrees; i++)
		{
			for (int j = 0; j < numIndices; j++)
			{
				posteriors[i * numIndices + j] = 0;
				positives[i * numIndices + j] = 0;
				negatives[i * numIndices + j] = 0;
			}
		}
	}



	void EnsembleClassifier::integralImag_extract(const Mat &src, Mat &sum, Mat &sqsum)//void VarianceFilter::integralImag_extract(const Mat &src)
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
		//	cout << "Error: src.data kernel_intgegral_cols_en EnqueueNDRangeKernel!" << endl;
		//}

		status = clSetKernelArg(kernel_intgegral_cols_en, 0, sizeof(cl_mem), (void *)&srcdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&src.data));
		status = clSetKernelArg(kernel_intgegral_cols_en, 1, sizeof(cl_mem), (void *)&tsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sum.data));
		status = clSetKernelArg(kernel_intgegral_cols_en, 2, sizeof(cl_mem), (void *)&tsqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sqsum.data));
		status = clSetKernelArg(kernel_intgegral_cols_en, 3, sizeof(cl_int), (void *)&offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&offset));
		status = clSetKernelArg(kernel_intgegral_cols_en, 4, sizeof(cl_int), (void *)&pre_invalid);//args.push_back(make_pair(sizeof(cl_int), (void *)&pre_invalid));
		status = clSetKernelArg(kernel_intgegral_cols_en, 5, sizeof(cl_int), (void *)&rows);//args.push_back(make_pair(sizeof(cl_int), (void *)&src.rows));
		status = clSetKernelArg(kernel_intgegral_cols_en, 6, sizeof(cl_int), (void *)&cols);//args.push_back(make_pair(sizeof(cl_int), (void *)&src.cols));
		status = clSetKernelArg(kernel_intgegral_cols_en, 7, sizeof(cl_int), &src_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&src.step));
		status = clSetKernelArg(kernel_intgegral_cols_en, 8, sizeof(cl_int), &t_sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.step));



		size_t gt[3] = { ((vcols + 1) / 2) * 256, 1, 1 }, lt[3] = { 256, 1, 1 };
		//	//openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_cols", gt, lt, args, -1, depth);
		status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_cols_en, 1, NULL, gt, lt, 0, NULL, &events[0]);
		status = clWaitForEvents(1, &events[0]);
		assert(status == CL_SUCCESS);
		//	
		//	status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_cols_en, 1, NULL, gt, lt, 0, NULL, NULL);
		//
		//if (status != CL_SUCCESS)
		//{
		//	cout << "Error: kernel_intgegral_cols_en EnqueueNDRangeKernel!" << endl;
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



		status = clSetKernelArg(kernel_intgegral_rows_en, 0, sizeof(cl_mem), (void *)&tsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 1, sizeof(cl_mem), (void *)&tsqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sqsum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 2, sizeof(cl_mem), (void *)&sumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&sum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 3, sizeof(cl_mem), (void *)&sqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&sqsum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 4, sizeof(cl_int), (void *)&t_sum.rows);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.rows));
		status = clSetKernelArg(kernel_intgegral_rows_en, 5, sizeof(cl_int), (void *)&t_sum.cols);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.cols));
		status = clSetKernelArg(kernel_intgegral_rows_en, 6, sizeof(cl_int), (void *)&t_sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.step));
		status = clSetKernelArg(kernel_intgegral_rows_en, 7, sizeof(cl_int), (void *)&sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&sum.step));
		status = clSetKernelArg(kernel_intgegral_rows_en, 8, sizeof(cl_int), (void *)&sqsum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&sqsum.step));
		status = clSetKernelArg(kernel_intgegral_rows_en, 9, sizeof(cl_int), (void *)&sum_offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&sum_offset));
		status = clSetKernelArg(kernel_intgegral_rows_en, 10, sizeof(cl_int), (void *)&sqsum_offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&sqsum_offset));

		size_t gt2[3] = { t_sum.cols * 32, 1, 1 }, lt2[3] = { 256, 1, 1 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_rows_en, 1, NULL, gt2, lt2, 0, NULL, &events[1]);//openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_rows", gt2, lt2, args, -1, depth);


																														//if (status != CL_SUCCESS)
																														//{
																														//	cout << "Error: kernel_intgegral_rows_en_en EnqueueNDRangeKernel!" << endl;
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
		//	}kernel_intgegral_cols_en
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
		//	clReleaseKernel(kernel_intgegral_rows_en);
		//	clReleaseKernel(kernel_intgegral_cols_en);
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

	void EnsembleClassifier::nextIteration(const Mat &img)
	{
		if (!enabled) return;

		this->img = (const unsigned char *)img.data;
	}

	//Classical fern algorithm
	int EnsembleClassifier::calcFernFeature(int windowIdx, int treeIdx)
	{

		int index = 0;
		int *bbox = windowOffsets + windowIdx * TLD_WINDOW_OFFSET_SIZE;
		int *off = featureOffsets + bbox[4] + treeIdx * 2 * numFeatures; //bbox[4] is pointer to features for the current scale

		for (int i = 0; i < numFeatures; i++)
		{
			index <<= 1;

			int fp0 = img[bbox[0] + off[0]];
			int fp1 = img[bbox[0] + off[1]];

			if (fp0 > fp1)
			{
				index |= 1;
			}

			off += 2;
		}

		return index;
	}

	void EnsembleClassifier::calcFeatureVector(int windowIdx, int *featureVector)
	{
		for (int i = 0; i < numTrees; i++)
		{
			featureVector[i] = calcFernFeature(windowIdx, i);
		}
	}

	float EnsembleClassifier::calcConfidence(int *featureVector)
	{
		float conf = 0.0;
		int temp = 0;
		for (int i = 0; i < numTrees; i++)
		{
			temp = featureVector[i];
			conf += posteriors[i * numIndices + temp];
			//conf += posteriors[i * numIndices + featureVector[i]];
		}

		return conf;
	}

	void EnsembleClassifier::classifyWindow2(int windowIdx)
	{
		float conf = 0.0;
		int *featureVector = detectionResult->featureVectors + numTrees * windowIdx;
		calcFeatureVector(windowIdx, featureVector);
		conf = calcConfidence(featureVector);
		detectionResult->posteriors[windowIdx] = conf;

	}

	bool EnsembleClassifier::filter(int i)
	{
		if (!enabled) return true;

		classifyWindow(i);

		if (detectionResult->posteriors[i] < 0.5) return false;

		return true;
	}
	void EnsembleClassifier::classifyWindow(int windowIdx)
	{
		int index, *off, *bbox;
		float conf = 0.0;
		int treeIdx = 0;
		int temp1, temp2;
		int *featureVector = detectionResult->featureVectors + numTrees * windowIdx;
		int idx = 0;
		for (treeIdx = 0; treeIdx < numTrees; treeIdx++)
		{

			index = 0;

			bbox = windowOffsets + windowIdx * TLD_WINDOW_OFFSET_SIZE;
			off = featureOffsets + bbox[4] + treeIdx * 2 * numFeatures; //bbox[4] is pointer to features for the current scale

			for (int i = 0; i < numFeatures; i++)
			{
				index <<= 1;

				int fp0 = img[bbox[0] + off[0]];
				int fp1 = img[bbox[0] + off[1]];

				if (fp0 > fp1)
				{
					index |= 1;
				}

				off += 2;
				idx++;
				//if (windowIdx == 10000)
				//	printf("idx=%d\t fp0=%d,fp1=%d, bbox[0]=%d,off[0]=%d\t,bbox +off=%d,index=%d ***\n",idx, fp0, fp1, bbox[0], off[0], bbox[0] + off[0],index);
			}

			featureVector[treeIdx] = index;


			conf += posteriors[treeIdx * numIndices + featureVector[treeIdx]];
			//if (windowIdx==753666)
			//{
			//	//printf("idx=%d\t,conf=%f,posterirors[]=%f\n", windowIdx, conf, posteriors[treeIdx * numIndices + featureVector[treeIdx]]); Sleep(1000);
			//	printf("windowIdx=%d,featureVector[treeIdx=%d]=%d,conf=%f   cpu\n", windowIdx, treeIdx, featureVector[treeIdx],conf); 
			//	printf("treeIdx * numIndices + index=%d,posteriors[ xx] = %f.............\n", treeIdx * numIndices + index, posteriors[treeIdx * numIndices + index]); 
			//	Sleep(1000);
			//}


		}

		detectionResult->posteriors[windowIdx] = conf;
		//if (conf >= 0.5)
		//	{
		//		printf("idx=%d\t,conf=%f\n", windowIdx, conf); Sleep(1000);
		//	}
	}



	bool EnsembleClassifier::clfilter(const Mat &matImg)
	{
		/*******  ******/
		/*Begain Integral */
		/******  *******/
		int i, j;
		cv::Mat t_sum;	//cv::Mat sum;
		cv::Mat t_sqsum;//cv::Mat sqsum;

		int vlen = 4; cl_event events[10];
		int offset = 0;
		int pre_invalid = 0;
		int vcols = (pre_invalid + matImg.cols + vlen - 1) / vlen;

		int w = matImg.cols, h = matImg.rows;
		int depth = matImg.depth() == CV_8U ? CV_32S : CV_64F;
		int type = CV_MAKE_TYPE(depth, 1);


		t_sum.create(matImg.cols, matImg.rows, type);
		iSumMat_ensemble.create(h, w, type);

		t_sqsum.create(matImg.cols, matImg.rows, CV_32FC1);
		fSqreSumMat_ensemble.create(h, w, CV_32FC1);

		int sum_offset = 0;    // sum.offset / vlen;
		int sqsum_offset = 0;  // sqsum.offset / vlen;


		cl_mem srcdata = clCreateBuffer(context, CL_MEM_READ_ONLY, (matImg.cols) * (matImg.rows) * sizeof(uchar), NULL, NULL);
		cl_mem tsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (matImg.cols) * (matImg.rows) * sizeof(int), NULL, NULL);
		cl_mem tsqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (matImg.cols) * (matImg.rows) * sizeof(float), NULL, NULL);
		cl_mem sumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (matImg.cols) * (matImg.rows) * sizeof(int), NULL, NULL);
		cl_mem sqsumdata = clCreateBuffer(context, CL_MEM_READ_WRITE, (matImg.cols) * (matImg.rows) * sizeof(float), NULL, NULL);

		int step;
		int src_step, sum_step;
		int t_sum_step, sqsum_step;


		int rows = matImg.rows;
		int cols = matImg.cols;
		src_step = matImg.step;
		t_sum_step = t_sum.step;
		sum_step = iSumMat_ensemble.step;
		sqsum_step = fSqreSumMat_ensemble.step;

		//printf("");
		status = clEnqueueWriteBuffer(commandQueue, srcdata, CL_FALSE, 0, (matImg.cols) * (matImg.rows) * sizeof(uchar), matImg.data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(kernel_intgegral_cols_en, 0, sizeof(cl_mem), (void *)&srcdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&matImg.data));
		status = clSetKernelArg(kernel_intgegral_cols_en, 1, sizeof(cl_mem), (void *)&tsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sum.data));
		status = clSetKernelArg(kernel_intgegral_cols_en, 2, sizeof(cl_mem), (void *)&tsqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sqsum.data));
		status = clSetKernelArg(kernel_intgegral_cols_en, 3, sizeof(cl_int), (void *)&offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&offset));
		status = clSetKernelArg(kernel_intgegral_cols_en, 4, sizeof(cl_int), (void *)&pre_invalid);//args.push_back(make_pair(sizeof(cl_int), (void *)&pre_invalid));
		status = clSetKernelArg(kernel_intgegral_cols_en, 5, sizeof(cl_int), (void *)&rows);//args.push_back(make_pair(sizeof(cl_int), (void *)&matImg.rows));
		status = clSetKernelArg(kernel_intgegral_cols_en, 6, sizeof(cl_int), (void *)&cols);//args.push_back(make_pair(sizeof(cl_int), (void *)&matImg.cols));
		status = clSetKernelArg(kernel_intgegral_cols_en, 7, sizeof(cl_int), &src_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&matImg.step));
		status = clSetKernelArg(kernel_intgegral_cols_en, 8, sizeof(cl_int), &t_sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.step));


		size_t gt[3] = { ((vcols + 1) / 2) * 256, 1, 1 }, lt[3] = { 256, 1, 1 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_cols_en, 1, NULL, gt, lt, 0, NULL, &events[0]);
		status = clWaitForEvents(1, &events[0]);
		assert(status == CL_SUCCESS);


		status = clSetKernelArg(kernel_intgegral_rows_en, 0, sizeof(cl_mem), (void *)&tsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 1, sizeof(cl_mem), (void *)&tsqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&t_sqsum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 2, sizeof(cl_mem), (void *)&sumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&sum.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 3, sizeof(cl_mem), (void *)&sqsumdata); //args.push_back(make_pair(sizeof(cl_mem), (void *)&fSqreSumMat_ensemble.data));
		status = clSetKernelArg(kernel_intgegral_rows_en, 4, sizeof(cl_int), (void *)&t_sum.rows);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.rows));
		status = clSetKernelArg(kernel_intgegral_rows_en, 5, sizeof(cl_int), (void *)&t_sum.cols);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.cols));
		status = clSetKernelArg(kernel_intgegral_rows_en, 6, sizeof(cl_int), (void *)&t_sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&t_sum.step));
		status = clSetKernelArg(kernel_intgegral_rows_en, 7, sizeof(cl_int), (void *)&sum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&sum.step));
		status = clSetKernelArg(kernel_intgegral_rows_en, 8, sizeof(cl_int), (void *)&sqsum_step);//args.push_back(make_pair(sizeof(cl_int), (void *)&fSqreSumMat_ensemble.step));
		status = clSetKernelArg(kernel_intgegral_rows_en, 9, sizeof(cl_int), (void *)&sum_offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&sum_offset));
		status = clSetKernelArg(kernel_intgegral_rows_en, 10, sizeof(cl_int), (void *)&sqsum_offset);//args.push_back(make_pair(sizeof(cl_int), (void *)&sqsum_offset));

		size_t gt2[3] = { t_sum.cols * 32, 1, 1 }, lt2[3] = { 256, 1, 1 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel_intgegral_rows_en, 1, NULL, gt2, lt2, 0, NULL, &events[1]);//openCLExecuteKernel(src.clCxt, &imgproc_integral, "integral_rows", gt2, lt2, args, -1, depth);
																														//if (status != CL_SUCCESS)
																														//{
																														//	cout << "Error: kernel_intgegral_rows_en_en EnqueueNDRangeKernel!" << endl;
																														//}
		status = clWaitForEvents(1, &events[1]);
		assert(status == CL_SUCCESS);
		t_sum.release();
		t_sqsum.release();
		clReleaseMemObject(tsumdata);
		clReleaseMemObject(tsqsumdata);
/*
 * End Integral  
 */

/* 
 * Begain variance && ensemble filter or Posterior  
 */

		if (!enabled) return true;
		int tld_window_offset_size = TLD_WINDOW_OFFSET_SIZE;

		oclbuffWindowsOffset = clCreateBuffer(context, CL_MEM_READ_ONLY, (TLD_WINDOW_OFFSET_SIZE * numWindows) * sizeof(int), NULL, NULL);
		oclbufffeatureOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY, (numScales * numTrees * numFeatures * 2) * sizeof(int), NULL, NULL);
		oclbuffDetectionResultfeatureVectors = clCreateBuffer(context, CL_MEM_READ_WRITE, (numWindows * numTrees)* sizeof(int), NULL, NULL);
		oclbuffDetectionResultPosteriors = clCreateBuffer(context, CL_MEM_READ_WRITE, (numWindows)* sizeof(float), NULL, NULL);
		oclbuffPosteriors = clCreateBuffer(context, CL_MEM_READ_WRITE, (numTrees * numIndices)* sizeof(float), NULL, NULL);
		oclbuffDetectionResultVarious = clCreateBuffer(context, CL_MEM_READ_WRITE, (numWindows)* sizeof(float), NULL, NULL);


		status = clEnqueueWriteBuffer(commandQueue, oclbuffWindowsOffset, CL_FALSE, 0, (TLD_WINDOW_OFFSET_SIZE * numWindows) * sizeof(int), (void *)windowOffsets, 0, NULL, NULL);
		status = clEnqueueWriteBuffer(commandQueue, oclbufffeatureOffsets, CL_FALSE, 0, (numScales * numTrees * numFeatures * 2) * sizeof(int), (void *)featureOffsets, 0, NULL, NULL);
		status = clEnqueueWriteBuffer(commandQueue, oclbuffDetectionResultfeatureVectors, CL_FALSE, 0, (numWindows * numTrees)* sizeof(int), (void *)(void *)detectionResult->featureVectors, 0, NULL, NULL);
		status = clEnqueueWriteBuffer(commandQueue, oclbuffDetectionResultPosteriors, CL_FALSE, 0, (numWindows)* sizeof(float), (void *)detectionResult->posteriors, 0, NULL, NULL);
		status = clEnqueueWriteBuffer(commandQueue, oclbuffPosteriors, CL_FALSE, 0, (numTrees * numIndices)* sizeof(float), (void *)posteriors, 0, NULL, NULL);
		status = clEnqueueWriteBuffer(commandQueue, oclbuffDetectionResultVarious, CL_FALSE, 0, (numWindows)* sizeof(float), (void *)detectionResult->variances, 0, NULL, NULL);


		status = clSetKernelArg(variance_ensemble_kernel, 0, sizeof(int), (void *)&numIndices);
		status = clSetKernelArg(variance_ensemble_kernel, 1, sizeof(int), (void *)&numTrees);
		status = clSetKernelArg(variance_ensemble_kernel, 2, sizeof(int), (void *)&numFeatures);
		status = clSetKernelArg(variance_ensemble_kernel, 3, sizeof(int), (void *)&tld_window_offset_size);
		status = clSetKernelArg(variance_ensemble_kernel, 4, sizeof(cl_mem), (void *)&oclbuffWindowsOffset);

		status = clSetKernelArg(variance_ensemble_kernel, 5, sizeof(cl_mem), (void *)&srcdata);
		status = clSetKernelArg(variance_ensemble_kernel, 6, sizeof(cl_mem), (void *)&oclbufffeatureOffsets);
		status = clSetKernelArg(variance_ensemble_kernel, 7, sizeof(cl_mem), (void *)&oclbuffDetectionResultfeatureVectors);
		status = clSetKernelArg(variance_ensemble_kernel, 8, sizeof(cl_mem), (void *)&oclbuffDetectionResultPosteriors);
		status = clSetKernelArg(variance_ensemble_kernel, 9, sizeof(cl_mem), (void *)&oclbuffPosteriors);

		status = clSetKernelArg(variance_ensemble_kernel, 10, sizeof(cl_mem), (void *)&sumdata);
		status = clSetKernelArg(variance_ensemble_kernel, 11, sizeof(cl_mem), (void *)&sqsumdata);
		status = clSetKernelArg(variance_ensemble_kernel, 12, sizeof(cl_mem), (void *)&oclbuffDetectionResultVarious);
		status = clSetKernelArg(variance_ensemble_kernel, 13, sizeof(float), (void *)&varianceFilter->minVar);
		status = clSetKernelArg(variance_ensemble_kernel, 14, sizeof(float), (void *)&numWindows);


		size_t global_work_size[1] = { numWindows };
		//		size_t  global_work_size[1] = { (numWindows-255)/256*256 };
		//size_t global_work_size[1] = { (numWindows + 16*256 -1 )/4096  * 256 };  16windows per thread.
		size_t local_work_size[1] = { 256 };

		status = clEnqueueNDRangeKernel(commandQueue, variance_ensemble_kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[2]);
		//status = clWaitForEvents(1, &events[2]);
		if (status != CL_SUCCESS)
		{

			cout << "Error: EnsembleClassifier EnqueueNDRangeKernel!" << endl;
			//return false;
		}

		status = clEnqueueReadBuffer(commandQueue, oclbuffDetectionResultfeatureVectors, CL_FALSE, 0, (numWindows * numTrees)* sizeof(int), (void *)(void *)detectionResult->featureVectors, 0, NULL, &events[3]);
		//status = clWaitForEvents(1, &events[3]);

		status = clEnqueueReadBuffer(commandQueue, oclbuffDetectionResultPosteriors, CL_FALSE, 0, (numWindows)* sizeof(float), (void *)detectionResult->posteriors, 0, NULL, &events[4]);
		//status = clWaitForEvents(1, &events[4]);
		status = clEnqueueReadBuffer(commandQueue, oclbuffPosteriors, CL_FALSE, 0, (numTrees * numIndices)* sizeof(float), (void *)posteriors, 0, NULL, &events[5]);
		//status = clWaitForEvents(1, &events[5]);
		status = clEnqueueReadBuffer(commandQueue, oclbuffDetectionResultVarious, CL_FALSE, 0, (numWindows)* sizeof(float), (void *)detectionResult->variances, 0, NULL, &events[6]);
		//status = clWaitForEvents(1, &events[6]);

		for (int i = 0; i < 6; i++)
			status = clReleaseEvent(events[i]);	//cout << "o" << endl;

/* 
*  End ensemble filter or Posterior
*/

		clReleaseMemObject(oclbuffWindowsOffset);
		clReleaseMemObject(oclbufffeatureOffsets);
		clReleaseMemObject(oclbuffDetectionResultfeatureVectors);
		clReleaseMemObject(oclbuffDetectionResultPosteriors);
		clReleaseMemObject(oclbuffPosteriors);
		clReleaseMemObject(oclbuffDetectionResultVarious);
		clReleaseMemObject(srcdata);
		clReleaseMemObject(sumdata);
		clReleaseMemObject(sqsumdata);

		return true;
	}

	void EnsembleClassifier::updatePosterior(int treeIdx, int idx, int positive, int amount)
	{
		int arrayIndex = treeIdx * numIndices + idx;
		(positive) ? positives[arrayIndex] += amount : negatives[arrayIndex] += amount;
		posteriors[arrayIndex] = ((float)positives[arrayIndex]) / (positives[arrayIndex] + negatives[arrayIndex]) / (float)numTrees;
	}

	void EnsembleClassifier::updatePosteriors(int *featureVector, int positive, int amount)
	{

		for (int i = 0; i < numTrees; i++)
		{

			int idx = featureVector[i];
			updatePosterior(i, idx, positive, amount);

		}
	}

	void EnsembleClassifier::learn(int *boundary, int positive, int *featureVector)
	{
		if (!enabled) return;

		float conf = calcConfidence(featureVector);

		//Update if positive patch and confidence < 0.5 or negative and conf > 0.5
		if ((positive && conf < 0.5) || (!positive && conf > 0.5))
		{
			updatePosteriors(featureVector, positive, 1);
		}

	}


} /* namespace tld */
