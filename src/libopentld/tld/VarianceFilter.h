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
* VarianceFilter.h
*
*  Created on: Nov 16, 2011
*      Author: Georg Nebehay
*/

#ifndef VARIANCEFILTER_H_
#define VARIANCEFILTER_H_

#include <opencv/cv.h>
#include  <CL/cl.h>

#include "IntegralImage.h"
#include "DetectionResult.h"
#include "switch.h"

namespace tld
{

	class VarianceFilter
	{


	public:
		oclMat clsum, clsqsum;


		//move here from top, which was set public from private. //edited by xqc
		IntegralImage<int>* integralImg;
		IntegralImage<float>* integralImg_squared;
		bool enabled;
		int *windowOffsets;

		DetectionResult *detectionResult;

		float minVar;
		//for opencl begin	
		int numWindows;
		cl_uint         numPlatforms;	//the NO. of platforms
		cl_platform_id  platform;	//the chosen platform
		cl_int	        status;
		cl_uint		    numDevices;


		cl_device_id    *devices;
		cl_context       context;
		cl_command_queue commandQueue;
		cl_program       program;
		cl_mem           inputBuffer;
		cl_mem           outputBuffer;
		cl_kernel        kernel, kernel_intgegral_cols, kernel_intgegral_rows;
		cv::Mat iSumMat;
		cv::Mat fSqreSumMat;




		//for opencl end
		VarianceFilter();
		virtual ~VarianceFilter();

		void release();
		void nextIteration(const cv::Mat &img);
		bool filter(int idx);

		bool clfilter(const Mat &img);
		float calcVariance(int *off);
		int VarianceFilter::convertToString(const char *filename, std::string& s);
		void integralImag_opencv(const Mat &img);

		void integralImag_extract(const Mat &src, Mat &iSumMat, Mat &fSqreSumMat);



	};

} /* namespace tld */
#endif /* VARIANCEFILTER_H_ */
