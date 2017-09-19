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
* NNClassifier.cpp
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#include "NNClassifier.h"

#include "DetectorCascade.h"
#include "TLDUtil.h"

using namespace std;
using namespace cv;

namespace tld
{




	NNClassifier::NNClassifier()
	{
		thetaFP = .5;
		thetaTP = .65;

		truePositives = new vector<NormalizedPatch>();
		falsePositives = new vector<NormalizedPatch>();
		candidatesToNNClassifyVector = new vector<nnClassifyStruct>(); // another method is to array 
		candidatesToNNClassifyIndexVector = new vector<int>();
		pcandidatesToNNClassifyIndexArray = NULL;
		pcandidatesToNNClassifyPatches = NULL;
		pNNResultsArray = NULL;
		kernel_nnClassifier = NULL;
		pSrcTruePostiveData = NULL;
		pSrcFalsePostiveData = NULL;
		//nnClassifyStructInstance = new  nnClassifyStruct;
	}

	NNClassifier::~NNClassifier()
	{
		release();

		delete truePositives;
		delete falsePositives;
		delete pNNResultsArray;
		delete pcandidatesToNNClassifyIndexArray;
	}

	void NNClassifier::release()
	{
		falsePositives->clear();
		truePositives->clear();
		candidatesToNNClassifyVector->clear();
		candidatesToNNClassifyIndexVector->clear();
	}

	void NNClassifier::VectorToArray()
	{ 

		int truePostiveSize = truePositives->size();
		int falsePositiveSize = falsePositives->size();
		
		pSrcTruePostiveData = new float[truePostiveSize * TLD_PATCH_SIZE *TLD_PATCH_SIZE];
		float *p = pSrcTruePostiveData;
		float *q;
		for (int i = 0; i < truePostiveSize; i++)
		{

			q = truePositives->at(i).values;
			for (int j = 0; j < TLD_PATCH_SIZE *TLD_PATCH_SIZE; j++)
				*p++ = *q++;

		}
	
		pSrcFalsePostiveData = new float[falsePositiveSize * TLD_PATCH_SIZE *TLD_PATCH_SIZE];
		p = pSrcTruePostiveData;
	 	for (int i = 0; i < truePostiveSize; i++)
		{

			q = truePositives->at(i).values;
			for (int j = 0; j < TLD_PATCH_SIZE *TLD_PATCH_SIZE; j++)
				*p++ = *q++;
		}
	
	
	
	}
	float NNClassifier::ncc(float *f1, float *f2)
	{
		double corr = 0;
		double norm1 = 0;
		double norm2 = 0;

		int size = TLD_PATCH_SIZE * TLD_PATCH_SIZE;

		for (int i = 0; i < size; i++)
		{
			corr += f1[i] * f2[i];
			norm1 += f1[i] * f1[i];
			norm2 += f2[i] * f2[i];
		}

		// normalization to <0,1>

		return (corr / sqrt(norm1 * norm2) + 1) / 2.0;
	}

	float NNClassifier::classifyPatch(NormalizedPatch *patch)
	{

		if (truePositives->empty())
		{
			return 0;
		}

		if (falsePositives->empty())
		{
			return 1;
		}
		//printf("..................\n");
		//printf("truePositives->size()=%d \t,falsePositives->size()=%d\n", truePositives->size(), falsePositives->size());
		//printf("..................\n");
		float ccorr_max_p = 0;

		//Compare patch to positive patches
		for (size_t i = 0; i < truePositives->size(); i++)
		{
			float ccorr = ncc(truePositives->at(i).values, patch->values);

			if (ccorr > ccorr_max_p)
			{
				ccorr_max_p = ccorr;
			}
		}

		float ccorr_max_n = 0;

		//Compare patch to negative patches
		for (size_t i = 0; i < falsePositives->size(); i++)
		{
			float ccorr = ncc(falsePositives->at(i).values, patch->values);

			if (ccorr > ccorr_max_n)
			{
				ccorr_max_n = ccorr;
			}
		}

		float dN = 1 - ccorr_max_n;
		float dP = 1 - ccorr_max_p;

		float distance = dN / (dN + dP);
		return distance;
	}

	float NNClassifier::classifyBB(const Mat &img, Rect *bb)
	{
		NormalizedPatch patch;

		tldExtractNormalizedPatchRect(img, bb, patch.values);
		return classifyPatch(&patch);

	}

	float NNClassifier::classifyWindow(const Mat &img, int windowIdx)
	{
		NormalizedPatch patch;

		int *bbox = &windows[TLD_WINDOW_SIZE * windowIdx];
		tldExtractNormalizedPatchBB(img, bbox, patch.values);

		return classifyPatch(&patch);
	}

	bool NNClassifier::filter(const Mat &img, int windowIdx)
	{
		if (!enabled) return true;

		float conf = classifyWindow(img, windowIdx);
		//printf("    windowIdx=%d   conf = %f....\n", windowIdx, conf);


		if (conf < thetaTP)
		{
			return false;
		}

		return true;
	}


	bool NNClassifier::clNNFilter(const cv::Mat &img) 
	{
		
		cl_event events[2];
	
		int truePostiveSize = truePositives->size();
		int falsePositiveSize = falsePositives->size();
		int CandidatesToNNClassifySize = candidatesToNNClassifyIndexVector->size();
		
		pcandidatesToNNClassifyPatches = new  float [TLD_PATCH_SIZE * TLD_PATCH_SIZE *candidatesToNNClassifyIndexVector->size()];
		float * pPatches = pcandidatesToNNClassifyPatches;
		float maxPositiveValue  ;
		float maxFalseValue;
		pNNResultsArray = new float[ (truePostiveSize + falsePositiveSize)*candidatesToNNClassifyIndexVector->size()];
		for (int i = 0; i < (truePostiveSize + falsePositiveSize)*candidatesToNNClassifyIndexVector->size(); i++)
			pNNResultsArray[i] = 0.0f;
		
		pcandidatesToNNClassifyIndexArray = new float[candidatesToNNClassifyIndexVector->size()];
		float *pIndexArray = pcandidatesToNNClassifyIndexArray;
		NormalizedPatch patch;
		for (int i = 0; i < candidatesToNNClassifyIndexVector->size(); i++)
		{
			*pIndexArray++ = candidatesToNNClassifyIndexVector->at(i);
			int *bbox = &windows[TLD_WINDOW_SIZE * i];
			tldExtractNormalizedPatchBB(img, bbox, patch.values);
			float *pdest = patch.values;
			for (int j = 0; j < TLD_PATCH_SIZE*TLD_PATCH_SIZE; j++)
			{
				*pPatches++ = *pdest++;
			}
		}
		//printf("..................\n");
		//printf("truePositives->size()=%d \t,falsePositives->size()=%d\n", truePositives->size(), falsePositives->size());
		//printf("..................\n");
			
		// case one
		if (truePositives->empty())
		{
			//--return 0;
			//++set all ccorr_max_p equale to 1.0;
			for (int i = 0, count = 0; i < this->candidatesToNNClassifyVector->size(); i++)
			{
				candidatesToNNClassifyVector->at(i).conf = 0.0f;
				//candidatesToNNClassifyVector->at(i).index = i;
				//candidatesToNNClassifyVector->at(i).flag = false;

			}
			goto EndofFuction;
		}
		// case two
		if (!truePositives->empty() && falsePositives->empty())
		{
			//++ set all ccorr_max_p equale to 1.0;
			for (int i = 0, count = 0; i < this->candidatesToNNClassifyVector->size(); i++)
			{
				candidatesToNNClassifyVector->at(i).conf = 1.0f;
				//candidatesToNNClassifyVector->at(i).index = candidatesToNNClassifyIndexVector->at(i);
				//candidatesToNNClassifyVector->at(i).flag = true;

			}
				
			goto EndofFuction;
		}
		//case three  TODO£º 
		//if (truePositives->empty() && !falsePositives->empty())

		//case four
		if (!truePositives->empty() && !falsePositives->empty())
		{
		 
			int tld_window_size = TLD_WINDOW_SIZE;
			//VectorToArray();
			// Begin Vector to Array 
			pSrcTruePostiveData = new float[truePostiveSize * TLD_PATCH_SIZE *TLD_PATCH_SIZE];
			float *p = pSrcTruePostiveData;
			float *q;
			for (int i = 0; i < truePostiveSize; i++)
			{
				q = truePositives->at(i).values;
				for (int j = 0; j < TLD_PATCH_SIZE *TLD_PATCH_SIZE; j++)
					*p++ = *q++;
			}
			pSrcFalsePostiveData = new float[falsePositiveSize * TLD_PATCH_SIZE *TLD_PATCH_SIZE];
			p = pSrcFalsePostiveData;
			for (int i = 0; i < falsePositiveSize; i++)
			{
				q = falsePositives->at(i).values;
				for (int j = 0; j < TLD_PATCH_SIZE *TLD_PATCH_SIZE; j++)
					*p++ = *q++;
			}
			
			// End Vector to Array 
	    	oclbufferpNNResultsArray  = clCreateBuffer(context, CL_MEM_READ_WRITE |CL_MEM_USE_HOST_PTR, (truePostiveSize + falsePositiveSize)*candidatesToNNClassifyIndexVector->size()* sizeof(float), (void*)pNNResultsArray, NULL);
			oclbufferSrcTruePostiveData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, truePostiveSize * TLD_PATCH_SIZE *TLD_PATCH_SIZE* sizeof(float), (void*)pSrcTruePostiveData, NULL);
			oclbufferSrcFalsePostiveData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, falsePositiveSize * TLD_PATCH_SIZE *TLD_PATCH_SIZE* sizeof(float), (void*)pSrcFalsePostiveData, NULL);
			oclbufferCandidatesToNNClassifyPatches = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR, TLD_PATCH_SIZE * TLD_PATCH_SIZE *candidatesToNNClassifyIndexVector->size() *sizeof(int), (void*)pcandidatesToNNClassifyPatches, NULL);
 
			/*Step 8: Create kernel object */
			/*Step 9: Sets Kernel arguments.*/
		 	status = clSetKernelArg(kernel_nnClassifier, 0, sizeof(cl_mem), (void *)&oclbufferpNNResultsArray);
			status = clSetKernelArg(kernel_nnClassifier, 1, sizeof(int), (void *)&truePostiveSize);
			status = clSetKernelArg(kernel_nnClassifier, 2, sizeof(int), (void *)&falsePositiveSize);
			status = clSetKernelArg(kernel_nnClassifier, 3, sizeof(int), (void *)&CandidatesToNNClassifySize);
			status = clSetKernelArg(kernel_nnClassifier, 4, sizeof(cl_mem), (void *)&oclbufferSrcTruePostiveData);
			status = clSetKernelArg(kernel_nnClassifier, 5, sizeof(cl_mem), (void *)&oclbufferSrcFalsePostiveData);
			status = clSetKernelArg(kernel_nnClassifier, 6, sizeof(cl_mem), (void *)&oclbufferCandidatesToNNClassifyPatches);
 
			//status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&TLD_WINDOW_OFFSET_SIZE);

			/*Step 10: Running the kernel.*/
			//printf("begore opencl kernel numWindows=%d\n", numWindows);
			size_t global_work_size[1] = { truePostiveSize+falsePositiveSize };
			size_t local_work_size[1] = { 256 };
			status = clEnqueueNDRangeKernel(commandQueue, kernel_nnClassifier, 1, NULL, global_work_size, NULL, 0, NULL, &events[0]);
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


			status = clEnqueueReadBuffer(commandQueue, oclbufferpNNResultsArray, CL_FALSE, 0, (truePostiveSize + falsePositiveSize)*candidatesToNNClassifyIndexVector->size()* sizeof(float), (void *)pNNResultsArray, 0, NULL, &events[0]);
			  status = clWaitForEvents(1, &events[0]);
			  if (status != CL_SUCCESS)
			  {
				  cout << "Error:NNclassify EnqueueNDRangeKernel!" << endl;
				  //return false;
			  }

			/*  printf("Reading data from GPU *************************\n");
			  for (int i = 0; i < (truePostiveSize + falsePositiveSize)*candidatesToNNClassifyIndexVector->size(); i++)
			  {
				  printf("candidatesToNNClassifyIndexVector->size()=%d£¬ detectionResult[%d] is %f\n", candidatesToNNClassifyIndexVector->size(),i, pNNResultsArray[i]);


			  }
			*/
			  for (int i = 0; i < candidatesToNNClassifyIndexVector->size();i++)
			  {
				   maxPositiveValue = pNNResultsArray[i];
				   maxFalseValue = pNNResultsArray[i+ truePostiveSize*candidatesToNNClassifyIndexVector->size()];
				 for (int j = 1; j < truePostiveSize; j++)
				 {
					 if ( maxPositiveValue < pNNResultsArray[j*candidatesToNNClassifyIndexVector->size()])
						 maxPositiveValue = pNNResultsArray[j*candidatesToNNClassifyIndexVector->size()];

				 }
				 for (int j = truePostiveSize; j < truePostiveSize+falsePositiveSize; j++)
				 {
					 if (maxFalseValue < pNNResultsArray[j*candidatesToNNClassifyIndexVector->size()])
						 maxFalseValue = pNNResultsArray[j*candidatesToNNClassifyIndexVector->size()];

				 }
				 float dN = 1 - maxFalseValue;
				 float dP = 1 - maxPositiveValue;

				 float conf = dN / (dN + dP);
				// if (conf > thetaTP)
				 //{
				//	 return false;
				 //}
				     candidatesToNNClassifyVector->at(i).conf = conf;
					 candidatesToNNClassifyVector->at(i).index = candidatesToNNClassifyIndexVector->at(i);
					 candidatesToNNClassifyVector->at(i).flag = true;

				 }



			//status = clReleaseEvent(events[0]);
			//status = clReleaseEvent(events[1]);

			//printf("end using GPU*************************\n");
			//for (int i = 0; i < numWindows; i++)
			//	if(detectionResult->windowFlags[i]==1)
			//	printf("detectionResult[%d] is %d\n", i,detectionResult->windowFlags[i]);

			/*Step 11: Read the cout put back to host memory.*/
			//status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, 12 * sizeof(char), output, 0, NULL, NULL);


			delete pSrcTruePostiveData;
			pSrcTruePostiveData = NULL;
			delete pSrcFalsePostiveData;
			pSrcFalsePostiveData = NULL;
			delete pNNResultsArray; 
			pNNResultsArray = NULL;
			delete pcandidatesToNNClassifyIndexArray;
			pcandidatesToNNClassifyIndexArray = NULL;
			delete pcandidatesToNNClassifyPatches;
			pcandidatesToNNClassifyPatches = NULL;


	//		clReleaseKernel(kernel_nnClassifier);
	//		clReleaseProgram(program);
	//		clReleaseMemObject(oclbufferSrcData);
		//	clReleaseMemObject(oclbufferWindows);
	//		clReleaseMemObject(oclbuffercandidatesToNNClassifyIndexArray);
			clReleaseMemObject(oclbufferpNNResultsArray);
			clReleaseMemObject(oclbufferCandidatesToNNClassifyPatches);
		 


			return true;


			/*end copy */


		 
		}









		float ccorr_max_p = 0;
	EndofFuction:
		return true;
	
	
	}

	void NNClassifier::learn(vector<NormalizedPatch> patches)
	{
		//TODO: Randomization might be a good idea here
		for (size_t i = 0; i < patches.size(); i++)
		{

			NormalizedPatch patch = patches[i];

			float conf = classifyPatch(&patch);

			if (patch.positive && conf <= thetaTP)
			{
				truePositives->push_back(patch);
			}

			if (!patch.positive && conf >= thetaFP)
			{
				falsePositives->push_back(patch);
			}
		}

	}


} /* namespace tld */
