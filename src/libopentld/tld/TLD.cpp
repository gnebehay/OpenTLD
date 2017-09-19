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
* TLD.cpp
*
*  Created on: Nov 17, 2011
*      Author: Georg Nebehay
*/

#include "TLD.h"

#include <iostream>
#include<Windows.h>
#include<windows.h>
#include "NNClassifier.h"
#include "TLDUtil.h"



#include <string.h>
#include "switch.h"
#include<fstream>


using namespace std;
using namespace cv;

namespace tld
{

	TLD::TLD()
	{
		trackerEnabled = true;
		detectorEnabled = true;
		learningEnabled = true;
		alternating = false;
		valid = false;
		wasValid = false;
		learning = false;
		currBB = NULL;
		prevBB = new Rect(0, 0, 0, 0);

		detectorCascade = new DetectorCascade();
		nnClassifier = detectorCascade->nnClassifier;

		//TLD tld opencl enviroments
		numWindows = detectorCascade->numWindows;
		platform = detectorCascade->platform;
		devices = detectorCascade->devices;
		context = detectorCascade->context;
		commandQueue = detectorCascade->commandQueue;



		medianFlowTracker = new MedianFlowTracker();

		int tld_window_size = TLD_WINDOW_SIZE;
		char *kernelName = "..\\..\\..\\src\\libopentld\\kernel\\cltldOverlapRect_kernel.cpp";

		string sourceStr;
		status = convertToString2(kernelName, sourceStr);
	 
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
			//return false;
		}
	}

	TLD::~TLD()
	{
		storeCurrentData();

		if (currBB)
		{
			delete currBB;
			currBB = NULL;
		}

		if (detectorCascade)
		{
			delete detectorCascade;
			detectorCascade = NULL;
		}

		if (medianFlowTracker)
		{
			delete medianFlowTracker;
			medianFlowTracker = NULL;
		}

		if (prevBB)
		{
			delete prevBB;
			prevBB = NULL;
		}
	}

	void TLD::release()
	{
		detectorCascade->release();
		medianFlowTracker->cleanPreviousData();

		if (currBB)
		{
			delete currBB;
			currBB = NULL;
		}
	}
	int TLD::convertToString2(const char *filename, std::string& s)
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

	void TLD::storeCurrentData()
	{
		prevImg.release();
		prevImg = currImg; //Store old image (if any)
		if (currBB)//Store old bounding box (if any)
		{
			prevBB->x = currBB->x;
			prevBB->y = currBB->y;
			prevBB->width = currBB->width;
			prevBB->height = currBB->height;
		}
		else
		{
			prevBB->x = 0;
			prevBB->y = 0;
			prevBB->width = 0;
			prevBB->height = 0;
		}

		detectorCascade->cleanPreviousData(); //Reset detector results
		medianFlowTracker->cleanPreviousData();

		wasValid = valid;
	}

	void TLD::selectObject(const Mat &img, Rect *bb)
	{

#if  opencl
		printf("*******************I am using OpenCL***********************\n");
#else
		printf("*******************I am using CPU***********************\n");
#endif
		//Delete old object
		detectorCascade->release();

		detectorCascade->objWidth = bb->width;
		detectorCascade->objHeight = bb->height;

		//Init detector cascade
		detectorCascade->init();

		currImg = img;
		if (currBB)
		{
			delete currBB;
			currBB = NULL;
		}
		currBB = tldCopyRect(bb);
		currConf = 1;
		valid = true;

		initialLearning();

	}

	void TLD::processImage(const Mat &img)
	{
		storeCurrentData();
		Mat grey_frame;

		double tic, toc;
		cvtColor(img, grey_frame, CV_BGR2GRAY);
		currImg = grey_frame; // Store new image , right after storeCurrentData();

		if (trackerEnabled)
		{
			medianFlowTracker->track(prevImg, currImg, prevBB);
		}

		if (detectorEnabled && (!alternating || medianFlowTracker->trackerBB == NULL))
		{
#if PrintTime_detect
			tic = cvGetTickCount();
#endif
#if  clDetect
			detectorCascade->cldetect(grey_frame);
#else
			detectorCascade->detect(grey_frame);
#endif		

#if PrintTime_detect
			toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

			toc = toc / 1000000;
			printf("*****the tldDect time used is %f************\n", toc);//float fps = 1 / toc;
#endif		

		}

#if PrintTime_fuseHypotheses
		tic = cvGetTickCount();
#endif


		fuseHypotheses();
#if PrintTime_fuseHypotheses
		toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

		toc = toc / 1000000;
		printf("*****the PrintTime_fuseHypotheses time  used is %f************\n", toc);//float fps = 1 / toc;
#endif
#if PrintTime_learn
		tic = cvGetTickCount();

#endif
		learn();
#if PrintTime_learn
		toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

		toc = toc / 1000000;
		printf("*****the PrintTime_learn time  used is %f************\n", toc);//float fps = 1 / toc;
#endif	

	}

	void TLD::fuseHypotheses()
	{
		Rect *trackerBB = medianFlowTracker->trackerBB;
		int numClusters = detectorCascade->detectionResult->numClusters;
		Rect *detectorBB = detectorCascade->detectionResult->detectorBB;

		if (currBB)
		{
			delete currBB;
			currBB = NULL;
		}
		currConf = 0;
		valid = false;

		float confDetector = 0;

		if (numClusters == 1)
		{
			confDetector = nnClassifier->classifyBB(currImg, detectorBB);
		}

		if (trackerBB != NULL)
		{
			float confTracker = nnClassifier->classifyBB(currImg, trackerBB);
			if (currBB)
			{
				delete currBB;
				currBB = NULL;
			}

			if (numClusters == 1 && confDetector > confTracker && tldOverlapRectRect(*trackerBB, *detectorBB) < 0.5)
			{

				currBB = tldCopyRect(detectorBB);
				currConf = confDetector;
				//printf("FrameNum=%d,currConf = confDetector=%f....\n", detectorCascade->FramNum, currConf);
			}
			else
			{
				currBB = tldCopyRect(trackerBB);
				currConf = confTracker;
				//printf("FrameNum=%d,currConf = confTracker = %f....\n", detectorCascade->FramNum, currConf);
				if (confTracker > nnClassifier->thetaTP)
				{
					valid = true;
				}
				else if (wasValid && confTracker > nnClassifier->thetaFP)
				{
					valid = true;
				}
			}
		}
		else if (numClusters == 1)
		{
			if (currBB)
			{
				delete currBB;
				currBB = NULL;
			}
			currBB = tldCopyRect(detectorBB);
			currConf = confDetector;
			//printf("FrameNum=%d,numClusters == 1   currConf = confTracker=%f....\n", detectorCascade->FramNum,currConf);
		}

		/*
		float var = CalculateVariance(patch.values, nn->patch_size*nn->patch_size);

		if(var < min_var) { //TODO: Think about incorporating this
		printf("%f, %f: Variance too low \n", var, classifier->min_var);
		valid = 0;
		}*/
	}
	void TLD::cltldOverlapRect_self(int *windows, int numWindows, Rect *boundary, float *overlap)
	{


		cl_event events[1];
		int tld_window_size = TLD_WINDOW_SIZE;

		int bb0, bb1, bb2, bb3;

		bb0 = boundary->x;
		bb1 = boundary->y;
		bb2 = boundary->width;
		bb3 = boundary->height;
		//float *overlap = new float[detectorCascade->numWindows];


		cl_mem oclbuffWindowsOffset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (TLD_WINDOW_SIZE * numWindows) * sizeof(int), (void *)windows, NULL);
		cl_mem oclbuffOverlap = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)overlap, NULL);
		kernel = clCreateKernel(program, "cltldOverlapRect", NULL);


		status = clSetKernelArg(kernel, 0, sizeof(int), (void *)&bb0);
		status = clSetKernelArg(kernel, 1, sizeof(int), (void *)&bb1);
		status = clSetKernelArg(kernel, 2, sizeof(int), (void *)&bb2);
		status = clSetKernelArg(kernel, 3, sizeof(int), (void *)&bb3);

		status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oclbuffWindowsOffset);
		status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&oclbuffOverlap);
		status = clSetKernelArg(kernel, 6, sizeof(int), (void *)&tld_window_size);


		/*Step 10: Running the kernel.*/
		//printf("begore opencl kernel numWindows=%d\n", numWindows);
		size_t global_work_size[1] = { numWindows };
		size_t local_work_size[1] = { 256 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[0]);
		if (status != CL_SUCCESS)
		{
			cout << "Error:cltldOverlapRect_self  EnqueueNDRangeKernel!" << endl;
			//return false;
		}
		status = clWaitForEvents(1, &events[0]);

		if (status != CL_SUCCESS)
		{
			printf("Error: Waiting for kernel run to finish.	(clWaitForEvents0)\n");

		}
		//cout << "o" << endl;
		status = clReleaseEvent(events[0]);

		clReleaseMemObject(oclbuffWindowsOffset);
		clReleaseMemObject(oclbuffOverlap);

		//return true;

	}
	void TLD::cltldOverlapRect(int *windows, int numWindows, Rect *boundary, float *overlap)
	{

		//printf("begin using initial clFilter*************************\n");
		cl_event events[1];
		int tld_window_size = TLD_WINDOW_SIZE;
		char *kernelName = "..\\..\\..\\src\\libopentld\\kernel\\cltldOverlapRect_kernel.cpp";
		string sourceStr;
		status = convertToString2(kernelName, sourceStr);
		const char *source = sourceStr.c_str();
		size_t sourceSize[] = { strlen(source) };
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

		/*Step 6: Build program. */
		status = clBuildProgram(program, 1, &devices[SELECTED_DEVICE_ID], NULL, NULL, NULL);
		//printf("mid  using EnsembleClassifier clFilter*************************\n");
		if (status != CL_SUCCESS)
		{
			cout << "overlap Error: Getting platforms!" << endl;
			//return false;
		}
		int bb0, bb1, bb2, bb3;

		bb0 = boundary->x;
		bb1 = boundary->y;
		bb2 = boundary->width;
		bb3 = boundary->height;
		//float *overlap = new float[detectorCascade->numWindows];


		//int *windows, int numWindows, Rect *boundary, float *overlap
		cl_mem oclbuffWindowsOffset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (TLD_WINDOW_SIZE * numWindows) * sizeof(int), (void *)windows, NULL);
		cl_mem oclbuffOverlap = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (numWindows)* sizeof(float), (void *)overlap, NULL);


		/*Step 8: Create kernel object */
		kernel = clCreateKernel(program, "cltldOverlapRect", NULL);
		/*Step 9: Sets Kernel arguments.*/
		status = clSetKernelArg(kernel, 0, sizeof(int), (void *)&bb0);
		status = clSetKernelArg(kernel, 1, sizeof(int), (void *)&bb1);
		status = clSetKernelArg(kernel, 2, sizeof(int), (void *)&bb2);
		status = clSetKernelArg(kernel, 3, sizeof(int), (void *)&bb3);

		status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&oclbuffWindowsOffset);
		status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&oclbuffOverlap);
		status = clSetKernelArg(kernel, 6, sizeof(int), (void *)&tld_window_size);


		/*Step 10: Running the kernel.*/
		//printf("begore opencl kernel numWindows=%d\n", numWindows);
		size_t global_work_size[1] = { numWindows };
		size_t local_work_size[1] = { 256 };
		status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &events[0]);
		if (status != CL_SUCCESS)
		{
			cout << "Error: EnqueueNDRangeKernel!" << endl;
			//return false;
		}
		status = clWaitForEvents(1, &events[0]);

		if (status != CL_SUCCESS)
		{
			printf("Error: Waiting for kernel run to finish.	(clWaitForEvents0)\n");

		}
		//cout << "o" << endl;
		status = clReleaseEvent(events[0]);

		clReleaseKernel(kernel);
		clReleaseProgram(program);

		clReleaseMemObject(oclbuffWindowsOffset);
		clReleaseMemObject(oclbuffOverlap);

		//return true;

	}

	void TLD::initialLearning()
	{
		learning = true; //This is just for display purposes

		DetectionResult *detectionResult = detectorCascade->detectionResult;
#if  clDetect
		detectorCascade->cldetect(currImg);
#else
		detectorCascade->detect(currImg);
#endif
		// 
		//This is the positive patch
		NormalizedPatch patch;
		tldExtractNormalizedPatchRect(currImg, currBB, patch.values);
		patch.positive = 1;

		float initVar = tldCalcVariance(patch.values, TLD_PATCH_SIZE * TLD_PATCH_SIZE);
		detectorCascade->varianceFilter->minVar = initVar / 2;


		float *overlap = new float[detectorCascade->numWindows];
		//   double tic = cvGetTickCount();
#if clOverlap
		cltldOverlapRect_self(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);
#else
		tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);
#endif
		//	double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

		//	toc = toc / 1000000;
		//	 printf("*****the tldOverlapRect time used is %f************\n", toc);//float fps = 1 / toc;

		//	printf("****************i am 248426 overlap[248426]= %f*****************\n", overlap[248426]);
		//	printf("****************i am 249404 overlap[248426]= %f*****************\n", overlap[249404]);
		//Add all bounding boxes with high overlap

		vector< pair<int, float> > positiveIndices;
		vector<int> negativeIndices;

		//First: Find overlapping positive and negative patches

		for (int i = 0; i < detectorCascade->numWindows; i++)
		{

			if (overlap[i] > 0.6)
			{
				positiveIndices.push_back(pair<int, float>(i, overlap[i]));

			}

			if (overlap[i] < 0.2)
			{
				float variance = detectionResult->variances[i];
				//printf("variance=%f\n", variance);

				if (!detectorCascade->varianceFilter->enabled || variance > detectorCascade->varianceFilter->minVar)   //TODO: This check is unnecessary if minVar would be set before calling detect.
				{
					//printf("variance=%f\n", variance);
					negativeIndices.push_back(i);

				}
			}
		}
		//cout << "xxxxnegativeIndices.size()xxx" << negativeIndices.size() << endl;
		//	FILE *fp  = fopen("C:\\Users\\kevin\\Desktop\\compResult\\opentld_opencl_overlap.txt", "a+");
		//	for (int i = 0; i < negativeIndices.size();i++)
		//fprintf(fp, "windowIdx=%d,overlap=%f\n", i, overlap[i]);
		//		fprintf(fp, "%d\t%f\n", i, negativeIndices.at(i));
		//	fclose(fp);

		sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

		vector<NormalizedPatch> patches;

		patches.push_back(patch); //Add first patch to patch list

		int numIterations = std::min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)

		for (int i = 0; i < numIterations; i++)
		{
			int idx = positiveIndices.at(i).first;
			//Learn this bounding box
			//TODO: Somewhere here image warping might be possible
			detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
		}

		srand(1); //TODO: This is not guaranteed to affect random_shuffle

		random_shuffle(negativeIndices.begin(), negativeIndices.end());

		//Choose 100 random patches for negative examples
		for (size_t i = 0; i < std::min<size_t>(100, negativeIndices.size()); i++)
		{
			int idx = negativeIndices.at(i);

			NormalizedPatch patch;
			tldExtractNormalizedPatchBB(currImg, &detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
			patch.positive = 0;
			patches.push_back(patch);
		}

		detectorCascade->nnClassifier->learn(patches);

		delete[] overlap;

	}

	//Do this when current trajectory is valid
	void TLD::learn()
	{
		if (!learningEnabled || !valid || !detectorEnabled)
		{
			learning = false;
			return;
		}

		learning = true;

		DetectionResult *detectionResult = detectorCascade->detectionResult;

		if (!detectionResult->containsValidData)
		{

#if PrintTime_detect
			double tic = cvGetTickCount();
#endif
#if  clDetect
			detectorCascade->cldetect(currImg);
#else
			detectorCascade->detect(currImg);
#endif

#if PrintTime_detect
			double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

			toc = toc / 1000000;
			printf("*****the tldDect time used is %f************\n", toc);//float fps = 1 / toc;
#endif
		}

		//This is the positive patch
		NormalizedPatch patch;
		tldExtractNormalizedPatchRect(currImg, currBB, patch.values);

		float *overlap = new float[detectorCascade->numWindows];

#if PrintTime_overlap
		double tic = cvGetTickCount();
#endif
#if clOverlap
		cltldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);
#else 
		tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);
#endif
#if PrintTime_overlap 
		double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

		toc = toc / 1000000;
		printf("*****the tldOverlapRect time used is %f************\n", toc);//float fps = 1 / toc;
#endif

																			 //Add all bounding boxes with high overlap

		vector<pair<int, float> > positiveIndices;
		vector<int> negativeIndices;
		vector<int> negativeIndicesForNN;

		//First: Find overlapping positive and negative patches

		for (int i = 0; i < detectorCascade->numWindows; i++)
		{

			if (overlap[i] > 0.6)
			{
				positiveIndices.push_back(pair<int, float>(i, overlap[i]));
				//FILE *fp = fopen("opentld_sort_overlap.txt", "a+");
				//fprintf(fp, "windowIdx=%d,overlap=%f\n", i, overlap[i]);
				//fclose(fp);
			}

			if (overlap[i] < 0.2)
			{
				if (!detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5)   //Should be 0.5 according to the paper
				{
					negativeIndices.push_back(i);
					negativeIndicesForNN.push_back(i);
				}

				//if(!detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5)
				//{
				//    negativeIndicesForNN.push_back(i);  
				//}

			}
		}

		sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

		vector<NormalizedPatch> patches;

		patch.positive = 1;
		patches.push_back(patch);
		//TODO: Flip


		int numIterations = std::min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)

		for (size_t i = 0; i < negativeIndices.size(); i++)
		{
			int idx = negativeIndices.at(i);
			//TODO: Somewhere here image warping might be possible
			detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], false, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
		}

		//TODO: Randomization might be a good idea
		for (int i = 0; i < numIterations; i++)
		{
			int idx = positiveIndices.at(i).first;
			//TODO: Somewhere here image warping might be possible
			detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
		}

		for (size_t i = 0; i < negativeIndicesForNN.size(); i++)
		{
			int idx = negativeIndicesForNN.at(i);

			NormalizedPatch patch;
			tldExtractNormalizedPatchBB(currImg, &detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
			patch.positive = 0;
			patches.push_back(patch);
		}

		detectorCascade->nnClassifier->learn(patches);
		//cout << "xxxxnegativeIndices.size()xxx ....." << negativeIndices.size() << "negativeIndicesForNN.size()=...;;" << negativeIndicesForNN.size() << endl;
		//cout << "NN has now " << detectorCascade->nnClassifier->truePositives->size() << " positives and " << detectorCascade->nnClassifier->falsePositives->size() << " negatives.\n";

		delete[] overlap;
		/*debug	float *p;
		p = detectorCascade->ensembleClassifier->posteriors;
		for (int i = 0; i < 10; i++)
		{
		for (int j = 0; j < 8192; j++)
		{

		if (p[i * 8192 + j] > 0.11)
		printf("the tree is %d, the ensembleClassifier posteriors is %f\n",i , p[i * 8192 + j]);
		}
		printf("xxxxxxxxxxxxxxxxxxxxxx       the tree is %d\n", i);
		}
		*/
	}

	typedef struct
	{
		int index;
		int P;
		int N;
	} TldExportEntry;

	void TLD::writeToFile(const char *path)
	{
		NNClassifier *nn = detectorCascade->nnClassifier;
		EnsembleClassifier *ec = detectorCascade->ensembleClassifier;

		FILE *file = fopen(path, "w");
		fprintf(file, "#Tld ModelExport\n");
		fprintf(file, "%d #width\n", detectorCascade->objWidth);
		fprintf(file, "%d #height\n", detectorCascade->objHeight);
		fprintf(file, "%f #min_var\n", detectorCascade->varianceFilter->minVar);
		fprintf(file, "%d #Positive Sample Size\n", nn->truePositives->size());



		for (size_t s = 0; s < nn->truePositives->size(); s++)
		{
			float *imageData = nn->truePositives->at(s).values;

			for (int i = 0; i < TLD_PATCH_SIZE; i++)
			{
				for (int j = 0; j < TLD_PATCH_SIZE; j++)
				{
					fprintf(file, "%f ", imageData[i * TLD_PATCH_SIZE + j]);
				}

				fprintf(file, "\n");
			}
		}

		fprintf(file, "%d #Negative Sample Size\n", nn->falsePositives->size());

		for (size_t s = 0; s < nn->falsePositives->size(); s++)
		{
			float *imageData = nn->falsePositives->at(s).values;

			for (int i = 0; i < TLD_PATCH_SIZE; i++)
			{
				for (int j = 0; j < TLD_PATCH_SIZE; j++)
				{
					fprintf(file, "%f ", imageData[i * TLD_PATCH_SIZE + j]);
				}

				fprintf(file, "\n");
			}
		}

		fprintf(file, "%d #numtrees\n", ec->numTrees);
		detectorCascade->numTrees = ec->numTrees;
		fprintf(file, "%d #numFeatures\n", ec->numFeatures);
		detectorCascade->numFeatures = ec->numFeatures;

		for (int i = 0; i < ec->numTrees; i++)
		{
			fprintf(file, "#Tree %d\n", i);

			for (int j = 0; j < ec->numFeatures; j++)
			{
				float *features = ec->features + 4 * ec->numFeatures * i + 4 * j;
				fprintf(file, "%f %f %f %f # Feature %d\n", features[0], features[1], features[2], features[3], j);
			}

			//Collect indices
			vector<TldExportEntry> list;

			for (int index = 0; index < pow(2.0f, ec->numFeatures); index++)
			{
				int p = ec->positives[i * ec->numIndices + index];

				if (p != 0)
				{
					TldExportEntry entry;
					entry.index = index;
					entry.P = p;
					entry.N = ec->negatives[i * ec->numIndices + index];
					list.push_back(entry);
				}
			}

			fprintf(file, "%d #numLeaves\n", list.size());

			for (size_t j = 0; j < list.size(); j++)
			{
				TldExportEntry entry = list.at(j);
				fprintf(file, "%d %d %d\n", entry.index, entry.P, entry.N);
			}
		}

		fclose(file);

	}

	void TLD::readFromFile(const char *path)
	{
		release();

		NNClassifier *nn = detectorCascade->nnClassifier;
		EnsembleClassifier *ec = detectorCascade->ensembleClassifier;

		FILE *file = fopen(path, "r");

		if (file == NULL)
		{
			printf("Error: Model not found: %s\n", path);
			exit(1);
		}

		int MAX_LEN = 255;
		char str_buf[255];
		fgets(str_buf, MAX_LEN, file); /*Skip line*/

		fscanf(file, "%d \n", &detectorCascade->objWidth);
		fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
		fscanf(file, "%d \n", &detectorCascade->objHeight);
		fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

		fscanf(file, "%f \n", &detectorCascade->varianceFilter->minVar);
		fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

		int numPositivePatches;
		fscanf(file, "%d \n", &numPositivePatches);
		fgets(str_buf, MAX_LEN, file); /*Skip line*/


		for (int s = 0; s < numPositivePatches; s++)
		{
			NormalizedPatch patch;

			for (int i = 0; i < 15; i++)   //Do 15 times
			{

				fgets(str_buf, MAX_LEN, file); /*Read sample*/

				char *pch;
				pch = strtok(str_buf, " \n");
				int j = 0;

				while (pch != NULL)
				{
					float val = atof(pch);
					patch.values[i * TLD_PATCH_SIZE + j] = val;

					pch = strtok(NULL, " \n");

					j++;
				}
			}

			nn->truePositives->push_back(patch);
		}

		int numNegativePatches;
		fscanf(file, "%d \n", &numNegativePatches);
		fgets(str_buf, MAX_LEN, file); /*Skip line*/


		for (int s = 0; s < numNegativePatches; s++)
		{
			NormalizedPatch patch;

			for (int i = 0; i < 15; i++)   //Do 15 times
			{

				fgets(str_buf, MAX_LEN, file); /*Read sample*/

				char *pch;
				pch = strtok(str_buf, " \n");
				int j = 0;

				while (pch != NULL)
				{
					float val = atof(pch);
					patch.values[i * TLD_PATCH_SIZE + j] = val;

					pch = strtok(NULL, " \n");

					j++;
				}
			}

			nn->falsePositives->push_back(patch);
		}

		fscanf(file, "%d \n", &ec->numTrees);
		detectorCascade->numTrees = ec->numTrees;
		fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

		fscanf(file, "%d \n", &ec->numFeatures);
		detectorCascade->numFeatures = ec->numFeatures;
		fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

		int size = 2 * 2 * ec->numFeatures * ec->numTrees;
		ec->features = new float[size];
		ec->numIndices = pow(2.0f, ec->numFeatures);
		ec->initPosteriors();

		for (int i = 0; i < ec->numTrees; i++)
		{
			fgets(str_buf, MAX_LEN, file); /*Skip line*/

			for (int j = 0; j < ec->numFeatures; j++)
			{
				float *features = ec->features + 4 * ec->numFeatures * i + 4 * j;
				fscanf(file, "%f %f %f %f", &features[0], &features[1], &features[2], &features[3]);
				fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
			}

			/* read number of leaves*/
			int numLeaves;
			fscanf(file, "%d \n", &numLeaves);
			fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

			for (int j = 0; j < numLeaves; j++)
			{
				TldExportEntry entry;
				fscanf(file, "%d %d %d \n", &entry.index, &entry.P, &entry.N);
				ec->updatePosterior(i, entry.index, 1, entry.P);
				ec->updatePosterior(i, entry.index, 0, entry.N);
			}
		}

		detectorCascade->initWindowsAndScales();
		detectorCascade->initWindowOffsets();

		detectorCascade->propagateMembers();

		detectorCascade->initialised = true;

		ec->initFeatureOffsets();

		fclose(file);
	}


} /* namespace tld */
