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
* DetectorCascade.cpp
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#include "DetectorCascade.h"

#include <algorithm>

#include "TLDUtil.h"

#include <omp.h>



#include <string>
#include<iostream>
#include<fstream>
#include <windows.h>
#define SUCCESS 0
#define FAILURE 1
using namespace std;

using namespace cv;

namespace tld
{

	//TODO: Convert this to a function
#define sub2idx(x,y,imgWidthStep) ((int) (floor((x)+0.5) + floor((y)+0.5)*(imgWidthStep)))

	DetectorCascade::DetectorCascade()
	{
		objWidth = -1; //MUST be set before calling init
		objHeight = -1; //MUST be set before calling init
		useShift = 1;
		imgHeight = -1;
		imgWidth = -1;

		shift = 0.1;
		minScale = -10;
		maxScale = 10;
		minSize = 25;
		imgWidthStep = -1;

		numTrees = 10;
		numFeatures = 13;
		numFrame = 0;

		initialised = false;
		oclSetup();
		char * kernelName = "..\\..\\..\\src\\libopentld\\kernel\\EnsembleClassifierFilter_kernel.cl.cpp";
		oclBuildKernel(kernelName);


		foregroundDetector = new ForegroundDetector();
		varianceFilter = new VarianceFilter();
		ensembleClassifier = new EnsembleClassifier();
		nnClassifier = new NNClassifier();
		clustering = new Clustering();

		detectionResult = new DetectionResult();
	}

	DetectorCascade::~DetectorCascade()
	{
		release();

		delete foregroundDetector;
		delete varianceFilter;
		delete ensembleClassifier;
		delete nnClassifier;
		delete detectionResult;
		delete clustering;
	}


	int DetectorCascade::convertToString(const char *filename, std::string& s)
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

	void DetectorCascade::init()
	{
		if (imgWidth == -1 || imgHeight == -1 || imgWidthStep == -1 || objWidth == -1 || objHeight == -1)
		{
			//printf("Error: Window dimensions not set\n"); //TODO: Convert this to exception
		}

		initWindowsAndScales();
		initWindowOffsets();

		propagateMembers();

		ensembleClassifier->init();

		initialised = true;
	}

	//TODO: This is error-prone. Better give components a reference to DetectorCascade?
	void DetectorCascade::propagateMembers()
	{
		detectionResult->init(numWindows, numTrees);

		varianceFilter->windowOffsets = windowOffsets;
		varianceFilter->numWindows = numWindows;
		ensembleClassifier->numWindows = numWindows;//added by xqc
		ensembleClassifier->windowOffsets = windowOffsets;
		ensembleClassifier->imgWidthStep = imgWidthStep;
		ensembleClassifier->numScales = numScales;
		ensembleClassifier->scales = scales;
		ensembleClassifier->numFeatures = numFeatures;
		ensembleClassifier->numTrees = numTrees;
		nnClassifier->windows = windows;
		nnClassifier->numWindows = numWindows;
		clustering->windows = windows;
		clustering->numWindows = numWindows;

		foregroundDetector->minBlobSize = minSize * minSize;

		foregroundDetector->detectionResult = detectionResult;
		varianceFilter->detectionResult = detectionResult;
		ensembleClassifier->detectionResult = detectionResult;
		ensembleClassifier->varianceFilter = varianceFilter;
		nnClassifier->detectionResult = detectionResult;
		clustering->detectionResult = detectionResult;



		//ocl
		varianceFilter->platform = platform;
		varianceFilter->devices = devices;
		varianceFilter->context = context;
		varianceFilter->commandQueue = commandQueue;
		varianceFilter->program = program;
		varianceFilter->kernel_intgegral_cols = kernel_intgegral_cols;

		varianceFilter->kernel_intgegral_rows = kernel_intgegral_rows;


		//
		ensembleClassifier->platform = platform;
		ensembleClassifier->devices = devices;
		ensembleClassifier->context = context;
		ensembleClassifier->commandQueue = commandQueue;
		ensembleClassifier->program = program;
		ensembleClassifier->variance_ensemble_kernel = kernel_ensemble;
		ensembleClassifier->kernel_intgegral_cols_en = kernel_intgegral_cols;

		ensembleClassifier->kernel_intgegral_rows_en = kernel_intgegral_rows;
		//kernel_variance = clCreateKernel(program, "varianceFilter", NULL);

		//nnClassifier
		nnClassifier->platform = platform;
		nnClassifier->devices = devices;
		nnClassifier->context = context;
		nnClassifier->commandQueue = commandQueue;
		nnClassifier->program = program;
		nnClassifier->kernel_nnClassifier = kernel_nnClassifier;



	}

	void DetectorCascade::release()
	{
		if (!initialised)
		{
			return; //Do nothing
		}

		initialised = false;

		foregroundDetector->release();
		ensembleClassifier->release();
		nnClassifier->release();

		clustering->release();

		numWindows = 0;
		numScales = 0;

		delete[] scales;
		scales = NULL;
		delete[] windows;
		windows = NULL;
		delete[] windowOffsets;
		windowOffsets = NULL;

		objWidth = -1;
		objHeight = -1;

		detectionResult->release();
		oclRelease();
	}

	void DetectorCascade::cleanPreviousData()
	{
		detectionResult->reset();
	}

	/* returns number of bounding boxes, bounding boxes, number of scales, scales
	* bounding boxes are stored in an array of size 5*numBBs using the format <x y w h scaleIndex>
	* scales are stored using the format <w h>
	*
	*/
	void DetectorCascade::initWindowsAndScales()
	{

		int scanAreaX = 1; // It is important to start with 1/1, because the integral images aren't defined at pos(-1,-1) due to speed reasons
		int scanAreaY = 1;
		int scanAreaW = imgWidth - 1;
		int scanAreaH = imgHeight - 1;

		int windowIndex = 0;

		scales = new Size[maxScale - minScale + 1];

		numWindows = 0;

		int scaleIndex = 0;

		for (int i = minScale; i <= maxScale; i++)
		{
			float scale = pow(1.2, i);
			int w = (int)objWidth * scale;
			int h = (int)objHeight * scale;
			int ssw, ssh;

			if (useShift)
			{
				ssw = max<float>(1, w * shift);
				ssh = max<float>(1, h * shift);
			}
			else
			{
				ssw = 1;
				ssh = 1;
			}

			if (w < minSize || h < minSize || w > scanAreaW || h > scanAreaH) continue;

			scales[scaleIndex].width = w;
			scales[scaleIndex].height = h;

			scaleIndex++;

			numWindows += floor((float)(scanAreaW - w + ssw) / ssw) * floor((float)(scanAreaH - h + ssh) / ssh);
		}

		numScales = scaleIndex;

		windows = new int[TLD_WINDOW_SIZE * numWindows];

		for (scaleIndex = 0; scaleIndex < numScales; scaleIndex++)
		{
			int w = scales[scaleIndex].width;
			int h = scales[scaleIndex].height;

			int ssw, ssh;

			if (useShift)
			{
				ssw = max<float>(1, w * shift);
				ssh = max<float>(1, h * shift);
			}
			else
			{
				ssw = 1;
				ssh = 1;
			}

			for (int y = scanAreaY; y + h <= scanAreaY + scanAreaH; y += ssh)
			{
				for (int x = scanAreaX; x + w <= scanAreaX + scanAreaW; x += ssw)
				{
					int *bb = &windows[TLD_WINDOW_SIZE * windowIndex];
					tldCopyBoundaryToArray<int>(x, y, w, h, bb);
					bb[4] = scaleIndex;

					windowIndex++;
				}
			}

		}

		assert(windowIndex == numWindows);
		printf("******** DetectorCascade::initWindowsAndScales()********************%d**********************************\n", numWindows);
	}

	//oclSetup
	int DetectorCascade::oclSetup()
	{
		/*Step1: Getting platforms and choose an available one.*/

		platform = NULL;	//the chosen platform
		status = clGetPlatformIDs(0, NULL, &numPlatforms);
		if (status != CL_SUCCESS)
		{
			cout << "DetectorCascade oclSetup Error: Getting platforms!" << endl;
			return 0;
		}

		/*For clarity, choose the first available platform. */
		if (numPlatforms > 0)
		{
			cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
			status = clGetPlatformIDs(numPlatforms, platforms, NULL);
			platform = platforms[0];
			free(platforms);
		}
		/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
		numDevices = 0;

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		cout << "numDevices::" << numDevices << endl;
		if (numDevices == 0)	//no GPU available.
		{
			cout << "No GPU device available." << endl;
			cout << "Choose CPU as default device." << endl;
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		}
		else
		{
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		}
		for (cl_uint i = 0; i < numDevices; ++i)
		{
			char deviceName[1024];
			status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceName),
				deviceName, NULL);

			std::cout << "Device " << i << " : " << deviceName
				<< " Device ID is " << devices[i] << std::endl;
		}

		/*Step 3: Create context.*/
		context = clCreateContext(NULL, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);

		/*Step 4: Creating command queue associate with the context.*/
		commandQueue = clCreateCommandQueue(context, devices[SELECTED_DEVICE_ID], 0, NULL);
	}
	int DetectorCascade::oclBuildKernel(const char *kernelName)
	{
		/*Step 5: Create program object */
		//const char *filename = "M:\\OpenTLD\\OpenTLD-master\\OpenTLD-master\\src\\opentld\\HelloWorld_Kernel.cl";


		string sourceStr;
		status = convertToString(kernelName, sourceStr);
		const char *source = sourceStr.c_str();
		size_t sourceSize[] = { strlen(source) };
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

		/*Step 6: Build program. */
		status = clBuildProgram(program, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);
		if (status != CL_SUCCESS)
		{
			size_t log_size;
			char* program_log;
			clGetProgramBuildInfo(program, devices[SELECTED_DEVICE_ID], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			program_log = (char*)malloc(log_size + 1);
			clGetProgramBuildInfo(program, devices[SELECTED_DEVICE_ID], CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
			printf("%s\n", program_log);
			free(program_log);


			cout << "DetectorCascade::oclBuildKernelError:clBuildProgram  !" << endl;
			return false;
		}

		kernel_variance = clCreateKernel(program, "varianceFilter", NULL);
		kernel_ensemble = clCreateKernel(program, "EnsembleClassifierFilter", NULL);
		kernel_intgegral_cols = clCreateKernel(program, "integral_cols_DD", NULL);
		kernel_intgegral_rows = clCreateKernel(program, "integral_rows_DD", NULL);
		kernel_nnClassifier = clCreateKernel(program, "nnClassifier", NULL); 
	}

	void DetectorCascade::oclRelease()
	{
		/*Step 12: Clean the resources.*/
		if (kernel != NULL)
			status = clReleaseKernel(kernel);				//Release kernel.
															//status = clReleaseProgram(program);				//Release the program object.
															//status = clReleaseMemObject(inputBuffer);		//Release mem object.
															//status = clReleaseMemObject(outputBuffer);
															//status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
															//status = clReleaseContext(context);				//Release context.
		if (devices != NULL)
		{
			free(devices);
			devices = NULL;
		}
	}

	//Creates offsets that can be added to bounding boxes
	//offsets are contained in the form delta11, delta12,... (combined index of dw and dh)
	//Order: scale->tree->feature
	void DetectorCascade::initWindowOffsets()
	{

		windowOffsets = new int[TLD_WINDOW_OFFSET_SIZE * numWindows];

		int *off = windowOffsets;

		int windowSize = TLD_WINDOW_SIZE;
		//#pragma omp parallel for num_threads(8)
		for (int i = 0; i < numWindows; i++)
		{
			int *window = windows + windowSize * i;
			*off++ = sub2idx(window[0] - 1, window[1] - 1, imgWidthStep); // x1-1,y1-1
			*off++ = sub2idx(window[0] - 1, window[1] + window[3] - 1, imgWidthStep); // x1-1,y2
			*off++ = sub2idx(window[0] + window[2] - 1, window[1] - 1, imgWidthStep); // x2,y1-1
			*off++ = sub2idx(window[0] + window[2] - 1, window[1] + window[3] - 1, imgWidthStep); // x2,y2
			*off++ = window[4] * 2 * numFeatures * numTrees; // pointer to features for this scale
			*off++ = window[2] * window[3]; //Area of bounding box
		}
	}


	void DetectorCascade::cldetect(const Mat &img)
	{
		//For every bounding box, the output is confidence, pattern, variance
		//numFrame++; printf("the num of the  frame id is  %d,numWindows=%d\n", numFrame, numWindows);

		detectionResult->reset();
		if (!initialised)
		{
			return;
		}

		//Prepare components
		foregroundDetector->nextIteration(img); //Calculates foreground
												//double tic = cvGetTickCount();
												//varianceFilter->nextIteration(img); //Calculates integral images
												//varianceFilter->filter2();

												//double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

												//toc = toc / 1000000;
												//printf("*****the integral time used is %f************\n", toc);//float fps = 1 / toc;
		ensembleClassifier->nextIteration(img);


		//varianceFilter->clfilter(img);
		ensembleClassifier->clfilter(img);

		//for (int i = 0; i < numWindows; i++)
		//{
		//printf(" && detectionResult->windowFlags[%d]=%d\n", i, detectionResult->windowFlags[i]);
		/*	if (detectionResult->windowFlags[i] == 0)
		{
		printf("xxxxxxxxxxxxxxx windowIdx = %d\nxxxxxxxxxxxxxxxxxx", i);
		Sleep(500);
		}*/

		//}
		//for (int i = 0; i < numWindows; i++)
		//{
		//	//printf("flag=1 windowsidx = %d,flag=%d\n", i, detectionResult->windowFlags[i]);
		//	if (detectionResult->variances != 0)
		//	{
		//		printf("haha ... windowsidx = %d, detectionResult->variances=%f\n", i, detectionResult->variances[i]);
		//		Sleep(1000);
		//	}
		//}

		//printf("detectionResult[753666]=%f\n",detectionResult->posteriors[753666]);
#if clNNClassifier
		for (int i = 0, count = 0; i < numWindows; i++)
		{
			if (detectionResult->posteriors[i] >= 0.5f)
			{
				//float values;
				//int index;
				//bool flag;
				nnClassifier->nnClassifyStructInstance.conf = 0.0f;
				nnClassifier->nnClassifyStructInstance.index = i;
				nnClassifier->nnClassifyStructInstance.flag =  false;
				nnClassifier->candidatesToNNClassifyVector->push_back(nnClassifier->nnClassifyStructInstance);

				nnClassifier->candidatesToNNClassifyIndexVector->push_back(i);
			}

		}
		if (nnClassifier->candidatesToNNClassifyIndexVector->size()==0)
			goto EndNNclassify;

		 
		nnClassifier->clNNFilter(img);
		 
	


		for (int i = 0; i < nnClassifier->candidatesToNNClassifyVector->size(); i++)
		{
			if (nnClassifier->candidatesToNNClassifyVector->at(i).conf  > nnClassifier->thetaTP)
				detectionResult->confidentIndices->push_back(nnClassifier->candidatesToNNClassifyVector->at(i).index);
		}
	
		
		
		
	


		//·µ»Ø
		nnClassifier->candidatesToNNClassifyIndexVector->clear();

		nnClassifier->candidatesToNNClassifyVector->clear();
		

#else
		for (int i = 0, count = 0; i < numWindows; i++)
		{
			//	printf("I am victor %d\n", i);
			//if (detectionResult->windowFlags[i] == 0)
			if (detectionResult->posteriors[i] >= 0.5f)
			{
				// printf("windowIDx= %d,count=%d\n", i, count++);
				//count++;
				//printf("Framenum = %d, I am victor %d,count=%d\n", FramNum, i, count);
				if (!nnClassifier->filter(img, i))
				{
					continue;
				}
				//count++;

				detectionResult->confidentIndices->push_back(i);

			}

	}


#endif	


		
		//printf("confident_size %d\n\n", detectionResult->confidentIndices->size());//zhaodc

		//Cluster
	EndNNclassify:
		clustering->clusterConfidentIndices();

		detectionResult->containsValidData = true;
	
	EndofCldetect:
		;

}
	void DetectorCascade::detect(const Mat &img)
	{
		//For every bounding box, the output is confidence, pattern, variance

		detectionResult->reset();

		if (!initialised)
		{
			return;
		}

		//Prepare components
		foregroundDetector->nextIteration(img); //Calculates foreground
		varianceFilter->nextIteration(img); //Calculates integral images
		ensembleClassifier->nextIteration(img);

#pragma omp parallel for

		for (int i = 0; i < numWindows; i++)
		{

			int *window = &windows[TLD_WINDOW_SIZE * i];

			if (foregroundDetector->isActive())
			{
				bool isInside = false;

				for (size_t j = 0; j < detectionResult->fgList->size(); j++)
				{

					int bgBox[4];
					tldRectToArray(detectionResult->fgList->at(j), bgBox);

					if (tldIsInside(window, bgBox))  //TODO: This is inefficient and should be replaced by a quadtree
					{
						isInside = true;
					}
				}

				if (!isInside)
				{
					detectionResult->posteriors[i] = 0;
					continue;
				}
			}

			if (!varianceFilter->filter(i))
			{
				detectionResult->posteriors[i] = 0;
				continue;
			}

			if (!ensembleClassifier->filter(i))
			{
				continue;
			}

			if (!nnClassifier->filter(img, i))
			{
				continue;
			}

			detectionResult->confidentIndices->push_back(i);


		}

		//Cluster
		clustering->clusterConfidentIndices();

		detectionResult->containsValidData = true;
	}

} /* namespace tld */
