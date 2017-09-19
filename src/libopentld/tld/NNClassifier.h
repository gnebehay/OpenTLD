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
 * NNClassifier.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#ifndef NNCLASSIFIER_H_
#define NNCLASSIFIER_H_

#include <vector>

#include <opencv/cv.h>
#include  <CL/cl.h>

#include "NormalizedPatch.h"
#include "DetectionResult.h"

namespace tld
{


class nnClassifyStruct
{
public:
float conf;
int index;
bool flag;
};
 

class NNClassifier
{
    float ncc(float *f1, float *f2);
	
public:
    bool enabled;

    int *windows;
	//Configurable members
 
	int numWindows;
    float thetaFP;
    float thetaTP;
    DetectionResult *detectionResult;
    std::vector<NormalizedPatch>* falsePositives;
    std::vector<NormalizedPatch>* truePositives;  
	float *pNNResultsArray;
	float *pcandidatesToNNClassifyIndexArray;
	float *pcandidatesToNNClassifyPatches;
	std::vector<int> *candidatesToNNClassifyIndexVector;
	std::vector <nnClassifyStruct> * candidatesToNNClassifyVector;

	nnClassifyStruct  nnClassifyStructInstance;

	 
    NNClassifier();
    virtual ~NNClassifier();

    void release();
    float classifyPatch(NormalizedPatch *patch);
    float classifyBB(const cv::Mat &img, cv::Rect *bb);
    float classifyWindow(const cv::Mat &img, int windowIdx);
    void learn(std::vector<NormalizedPatch> patches);
    bool filter(const cv::Mat &img, int windowIdx);
	
	
	
	//for opencl begin in NNclassifier
	bool clNNFilter(const cv::Mat &img);
	// variable members
	cl_platform_id  platform;	//the chosen platform
	cl_int	        status;
	cl_device_id    *devices;
	cl_context       context;
	cl_command_queue commandQueue;
	cl_program       program;


	void VectorToArray();
	/*
	* class NormalizedPatch
	* {
	* public:
	*	float values[TLD_PATCH_SIZE *TLD_PATCH_SIZE];
	*	bool positive;
	*};
	*/
	float *pSrcTruePostiveData; //float values[TLD_PATCH_SIZE *TLD_PATCH_SIZE]; 
	float *pSrcFalsePostiveData;

	cl_mem  oclbufferSrcData;
	cl_mem  oclbufferWindows;
	cl_mem  oclbuffercandidatesToNNClassifyIndexArray;
	cl_mem 	oclbufferpNNResultsArray;
	cl_mem oclbufferSrcTruePostiveData;  // device memory to SrcTruePostiveData
	cl_mem oclbufferSrcFalsePostiveData; // device memory to SrcFalsePostiveData
	cl_mem oclbufferCandidatesToNNClassifyPatches; // candidatest to NNClassify patches from host to devices.

	 
	cl_kernel        kernel_nnClassifier;


	//for opencl end in NNclassifier

};

} /* namespace tld */
#endif /* NNCLASSIFIER_H_ */
