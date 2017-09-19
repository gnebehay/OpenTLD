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
* EnsembleClassifier.h
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#ifndef ENSEMBLECLASSIFIER_H_
#define ENSEMBLECLASSIFIER_H_

#include <opencv/cv.h>
#include  <CL/cl.h>

namespace tld
{

	class EnsembleClassifier
	{
		const unsigned char *img;

		float calcConfidence(int *featureVector);
		int calcFernFeature(int windowIdx, int treeIdx);
		void calcFeatureVector(int windowIdx, int *featureVector);
		void updatePosteriors(int *featureVector, int positive, int amount);



	public:
		bool enabled;

		//Configurable members
		int numTrees;
		int numFeatures;
		int numWindows;

		int imgWidthStep;
		int numScales;
		cv::Size *scales;

		int *windowOffsets;
		int *featureOffsets;
		float *features;

		int numIndices;

		float *posteriors;
		int *positives;
		int *negatives;

		DetectionResult *detectionResult;
		VarianceFilter *varianceFilter;

		cv::Mat iSumMat_ensemble;
		cv::Mat fSqreSumMat_ensemble;

		EnsembleClassifier();
		virtual ~EnsembleClassifier();
		void init();
		void initFeatureLocations();
		void initFeatureOffsets();
		void initPosteriors();
		void release();
		void integralImag_extract(const Mat &src, Mat &iSumMat, Mat &fSqreSumMat);

		void nextIteration(const cv::Mat &img);
		void classifyWindow(int windowIdx);
		void updatePosterior(int treeIdx, int idx, int positive, int amount);
		void learn(int *boundary, int positive, int *featureVector);
		bool filter(int i);
		bool EnsembleClassifier::clfilter(const Mat &img);
		void classifyWindow2(int windowIdx);


		int convertToString(const char *filename, std::string& s);


		//for opencl begin

		cl_platform_id  platform;	//the chosen platform
		cl_int	        status;
		cl_device_id    *devices;
		cl_context       context;
		cl_command_queue commandQueue;
		cl_program       program;
		cl_mem           inputBuffer;
		cl_mem           outputBuffer;
		cl_kernel        variance_ensemble_kernel;
		cl_kernel  kernel_intgegral_cols_en, kernel_intgegral_rows_en;

		cl_mem oclbuffWindowsOffset;
		cl_mem oclbufffeatureOffsets;
		cl_mem oclbuffDetectionResultfeatureVectors;
		cl_mem oclbuffDetectionResultPosteriors;
		cl_mem oclbuffDetectionwindowFlags;
		cl_mem oclbuffPosteriors;
		cl_mem oclbuffDetectionResultVarious;
		cl_mem oclbuffImgData;
		cl_mem oclbuffII;
		cl_mem oclbuffIISqure;
		size_t local_work_size[1];



		//for opencl end

	};

} /* namespace tld */
#endif /* ENSEMBLECLASSIFIER_H_ */
