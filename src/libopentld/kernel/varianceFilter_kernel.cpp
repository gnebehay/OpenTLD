

#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void varianceFilter(
 __global int* oclbuffWindowsOffset,
 __global int* oclbuffII, 
 __global float* oclbuffIISqure,  
 __global float* oclbuffDetectionResultVarious,
__global float* oclbuffDetectionResultPosteriors,
const unsigned int numWindows,
const float minVar,
__global int* windowFlags)
{
	  int windowIdx = get_global_id(0);  
	 //__global  int *off  = &(oclbuffWindowsOffset[windowIdx*6]) ;
	 //__global  int *off  = oclbuffWindowsOffset;
	  float bboxvar=0.0;
	  float mX;
	  float mX2; 	 

  
	  int off0  =oclbuffWindowsOffset[ windowIdx*6 + 0 ];
	  int off1  =oclbuffWindowsOffset[ windowIdx*6 + 1 ];
	  int off2  =oclbuffWindowsOffset[ windowIdx*6 + 2 ];
	  int off3  =oclbuffWindowsOffset[ windowIdx*6 + 3 ];
	  int off4  =oclbuffWindowsOffset[ windowIdx*6 + 4 ];
	  int off5  =oclbuffWindowsOffset[ windowIdx*6 + 5 ];		
      mX  = ( oclbuffII[off3] - oclbuffII[off2] -oclbuffII[off1] + oclbuffII[off0]  )/(float) off5;
	  mX2 = ( oclbuffIISqure[off3] - oclbuffIISqure[off2] -oclbuffIISqure[off1] + oclbuffIISqure[off0]  )/(float) off5;

	 bboxvar = mX2-mX*mX;
	 oclbuffDetectionResultVarious[windowIdx] = bboxvar;

	if(bboxvar < minVar)
	{
		oclbuffDetectionResultPosteriors[windowIdx] = 0.0;
		windowFlags[windowIdx] = 1;
	}
	//windowFlags[windowIdx] = 100;
	
	
}
 

 