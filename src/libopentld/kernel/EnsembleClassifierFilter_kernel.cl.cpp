

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TLD_PATCH_SIZE 15


__kernel void nnClassifier(
	__global float * NNResultsArray,
	const unsigned int truePostiveSize,
	const unsigned int falsePositiveSize,
	const unsigned int CandidatesToNNClassifySize,
	__global float * truePostiveData,
	__global float * FalsePostiveData,
	__global float * CandidatesToNNClassifyPatches)
{

	int ThreadID = get_global_id(0); //ThreadID is one of truePostiveSize or falsePositiveSize
	
	__global  float *pSrc1, *pSrc2;
	int patchSize = TLD_PATCH_SIZE * TLD_PATCH_SIZE;
	int CandidatesSize = CandidatesToNNClassifySize;
	__global  float *conf;
	
	pSrc2 = CandidatesToNNClassifyPatches;  //TLD_PATCH_SIZE=15
	conf = NNResultsArray + ThreadID *CandidatesSize ;
	if (ThreadID < truePostiveSize && ThreadID < (truePostiveSize + falsePositiveSize))
	{
		pSrc1 = truePostiveData + ThreadID * patchSize;  //TLD_PATCH_SIZE=15
		for (int j = 0; j < CandidatesSize; j++)
		{
			double corr = 0;
			double norm1 = 0;
			double norm2 = 0;
			for (int i = 0; i < patchSize; i++)
			{
				corr += pSrc1[i] * pSrc2[CandidatesSize*j + i];
				norm1 += pSrc1[i] * pSrc1[i];
				norm2 += pSrc2[j*CandidatesSize + i] * pSrc2[CandidatesSize*j + i];
			}
			// normalization to <0,1> 	//return (corr / sqrt(norm1 * norm2) + 1) / 2.0;

			if (norm1 == 0.0 || norm2 == 0.0)
			{
				norm1 = 1.0;
				norm2 = 1.0;
			}
			else 
				*conf++ = (corr / sqrt(norm1 * norm2) + 1) / 2.0;
		}
	}
	else if(ThreadID > truePostiveSize && ThreadID < (truePostiveSize + falsePositiveSize))
	{
		pSrc1 = FalsePostiveData+ (ThreadID- truePostiveSize) * patchSize;  //TLD_PATCH_SIZE=15
		for (int j = 0; j < CandidatesSize; j++)
		{
			double corr = 0;
			double norm1 = 0;
			double norm2 = 0;
			for (int i = 0; i < patchSize; i++)
			{
				corr += pSrc1[i] * pSrc2[i];
				norm1 += pSrc1[i] * pSrc1[i];
				norm2 += pSrc2[i] * pSrc2[i];
			}
			// normalization to <0,1>  //return (corr / sqrt(norm1 * norm2) + 1) / 2.0;
			if(norm1==0.0 || norm2 ==0.0)
			{
				norm1 = 1.0;
				norm2 = 1.0;
			}	  
			else
			*conf++ = (corr / sqrt(norm1 * norm2) + 1) / 2.0;

		}
 	}
	// tld_window_size = 5
 
	//resize(image(source(x, y, width, height), 15 * 15, patchValue);  //opencv resize function with bilinear method.
	
}


__kernel void EnsembleClassifierFilter(
    const int numIndices,
    const unsigned int numTrees,
    const unsigned int numFeatures,
    const unsigned int TLD_WINDOW_OFFSET_SIZE,
	__global int* oclbuffWindowsOffset,
    __global uchar * img,
	__global int* featureOffsets,
    __global float* oclbuffDetectionResultfeatureVectors, 
    __global float* oclbuffDetectionResultPosteriors,
    __global float* posteriors,
	 __global int* oclbuffII, 
    __global float* oclbuffIISqure,
		__global float* oclbuffDetectionResultVarious,
	const  float minVar,
    const   int  numWindows

    )
{
	__global int *off,  *bbox;
	__global int * featureVector;
	 	int bbx1,bbx2,bbx3,bbx4;


	int index,bbox4 ;
	float conf=0.0;
	int fp0,fp1;
	 int windowIdx = get_global_id(0); 
	 int gsize = get_global_size(0);
	if(windowIdx<numWindows)
	{
	 //Begin  variance
	 float bboxvar=0.0 ;
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
	 // if(windowIdx==160000)
	 //  printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  I am gidx=%d\t bboxvar=%f\t,gsize=%d",windowIdx,bboxvar,gsize);

	if(bboxvar < minVar)
	{
		oclbuffDetectionResultPosteriors[windowIdx] = 0.0;
	//	windowFlags[windowIdx] = 1;
		goto Endofkernel;
	}
	 //End variance

	featureVector =  oclbuffDetectionResultfeatureVectors + numTrees * windowIdx;
	bbox = oclbuffWindowsOffset + windowIdx * TLD_WINDOW_OFFSET_SIZE;
	bbox4 = bbox[4];
	int idx=0;
//	#pragma unroll
	for(int treeIdx = 0; treeIdx < numTrees; treeIdx++)
    {
        
		index = 0;
		 
		 
		off = featureOffsets + bbox[4] + treeIdx * 2 * numFeatures; //bbox[4] is pointer to features for the current scale

		for (int i = 0; i < numFeatures; i++)
		{
			index <<= 1;
			fp0 = (int)img[bbox[0] + off[0]];
			fp1 = (int)img[bbox[0] + off[1]];
				bbx1 = bbox[0];
			bbx2 = off[0];
			bbx3 = (int)img[86440];
			if (fp0 > fp1)
			{
				index |= 1;
			}
			off += 2;
		}
		 
		featureVector[treeIdx] = index;

		conf = conf+ posteriors[treeIdx * numIndices + index];	
		  	
    }
	 
	oclbuffDetectionResultPosteriors[windowIdx] = conf;
//	if(conf<0.5 )
//		windowFlags[windowIdx] = 2;
//	else
//		windowFlags[windowIdx] = 0;

	  // if(windowIdx==753666)
	   	// printf("conf=%f,windowFlags[windowIdx]=%d...... \n",conf,windowFlags[windowIdx]);
	//	if(conf>0.5f)
	  
Endofkernel:
	;
	}	
}	
 	 
#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define LSIZE 256
#define LSIZE_1 255
#define LSIZE_2 254
#define HF_LSIZE 128
#define LOG_LSIZE 8
#define LOG_NUM_BANKS 5
#define NUM_BANKS 32
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)


kernel void integral_cols_DD(__global uchar4 *src,__global int *sum ,__global float *sqsum,
                          int src_offset,int pre_invalid,int rows,int cols,int src_step,int dst_step)
{
 
// printf("src_offset = %d\n pre_invalid=%d\n,rows=%d\n,cols=%d\n,src_step=%d,dst_step=%d\n",src_offset,pre_invalid,rows,cols,src_step,dst_step);
  
     int lid = get_local_id(0);
    int gid = get_group_id(0);
    int4 src_t[2], sum_t[2];
    float4 sqsum_t[2];
    __local int4 lm_sum[2][LSIZE + LOG_LSIZE];
    __local float4 lm_sqsum[2][LSIZE + LOG_LSIZE];
    __local int* sum_p;
    __local float* sqsum_p;
    src_step = src_step >> 2;
    gid = gid << 1;
    for(int i = 0; i < rows; i =i + LSIZE_1)
    {
        src_t[0] = (i + lid < rows ? convert_int4(src[src_offset + (lid+i) * src_step + min(gid, cols - 1)]) : 0);
        src_t[1] = (i + lid < rows ? convert_int4(src[src_offset + (lid+i) * src_step + min(gid + 1, cols - 1)]) : 0);

        sum_t[0] = (i == 0 ? 0 : lm_sum[0][LSIZE_2 + LOG_LSIZE]);
        sqsum_t[0] = (i == 0 ? (float4)0 : lm_sqsum[0][LSIZE_2 + LOG_LSIZE]);
        sum_t[1] =  (i == 0 ? 0 : lm_sum[1][LSIZE_2 + LOG_LSIZE]);
        sqsum_t[1] =  (i == 0 ? (float4)0 : lm_sqsum[1][LSIZE_2 + LOG_LSIZE]);
        barrier(CLK_LOCAL_MEM_FENCE);

        int bf_loc = lid + GET_CONFLICT_OFFSET(lid);
        lm_sum[0][bf_loc] = src_t[0];
        lm_sqsum[0][bf_loc] = convert_float4(src_t[0] * src_t[0]);

        lm_sum[1][bf_loc] = src_t[1];
        lm_sqsum[1][bf_loc] = convert_float4(src_t[1] * src_t[1]);

        int offset = 1;
        for(int d = LSIZE >> 1 ;  d > 0; d>>=1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ai = offset * (((lid & 127)<<1) +1) - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi]  +=  lm_sum[lid >> 7][ai];
                lm_sqsum[lid >> 7][bi]  +=  lm_sqsum[lid >> 7][ai];
            }
            offset <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < 2)
        {
            lm_sum[lid][LSIZE_2 + LOG_LSIZE] = 0;
            lm_sqsum[lid][LSIZE_2 + LOG_LSIZE] = 0;
        }
        for(int d = 1;  d < LSIZE; d <<= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset >>= 1;
            int ai = offset * (((lid & 127)<<1) +1) - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi] += lm_sum[lid >> 7][ai];
                lm_sum[lid >> 7][ai] = lm_sum[lid >> 7][bi] - lm_sum[lid >> 7][ai];

                lm_sqsum[lid >> 7][bi] += lm_sqsum[lid >> 7][ai];
                lm_sqsum[lid >> 7][ai] = lm_sqsum[lid >> 7][bi] - lm_sqsum[lid >> 7][ai];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int loc_s0 = gid * dst_step + i + lid - 1 - pre_invalid * dst_step / 4, loc_s1 = loc_s0 + dst_step ;
        if(lid > 0 && (i+lid) <= rows)
        {
            lm_sum[0][bf_loc] += sum_t[0];
            lm_sum[1][bf_loc] += sum_t[1];
            lm_sqsum[0][bf_loc] += sqsum_t[0];
            lm_sqsum[1][bf_loc] += sqsum_t[1];
            sum_p = (__local int*)(&(lm_sum[0][bf_loc]));
            sqsum_p = (__local float*)(&(lm_sqsum[0][bf_loc]));
            for(int k = 0; k < 4; k++)
            {
                if(gid * 4 + k >= cols + pre_invalid || gid * 4 + k < pre_invalid) continue;
                sum[loc_s0 + k * dst_step / 4] = sum_p[k];
                sqsum[loc_s0 + k * dst_step / 4] = sqsum_p[k];
			
            }
            sum_p = (__local int*)(&(lm_sum[1][bf_loc]));
            sqsum_p = (__local float*)(&(lm_sqsum[1][bf_loc]));
            for(int k = 0; k < 4; k++)
            {
                if(gid * 4 + k + 4 >= cols + pre_invalid) break;
                sum[loc_s1 + k * dst_step / 4] = sum_p[k];
                sqsum[loc_s1 + k * dst_step / 4] = sqsum_p[k];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
			//	if(lid==0 && gid==0)
			//	printf("sum_start=%d,sqsum_start=%f\n",sum[lid],sqsum[lid]  );

}


kernel void integral_rows_DD(__global int4 *srcsum,__global float4 * srcsqsum,__global int *sum ,
                          __global float *sqsum,int rows,int cols,int src_step,int sum_step,
                          int sqsum_step,int sum_offset,int sqsum_offset)
{
 
     int lid = get_local_id(0);
    int gid = get_group_id(0);
    int4 src_t[2], sum_t[2];
    float4 sqsrc_t[2],sqsum_t[2];
    __local int4 lm_sum[2][LSIZE + LOG_LSIZE];
    __local float4 lm_sqsum[2][LSIZE + LOG_LSIZE];
    __local int *sum_p;
    __local float *sqsum_p;
	  			
				
	//			if(lid==0 && gid==0)
	//			printf("sum_p=%d,sqsum_p=%f\n",srcsum[0].x,srcsqsum[0].x  );

    src_step = src_step >> 4;
    for(int i = 0; i < rows; i =i + LSIZE_1)
    {
        src_t[0] = i + lid < rows ? srcsum[(lid+i) * src_step + gid * 2] : (int4)0;
        sqsrc_t[0] = i + lid < rows ? srcsqsum[(lid+i) * src_step + gid * 2] : (float4)0;
        src_t[1] = i + lid < rows ? srcsum[(lid+i) * src_step + gid * 2 + 1] : (int4)0;
        sqsrc_t[1] = i + lid < rows ? srcsqsum[(lid+i) * src_step + gid * 2 + 1] : (float4)0;

        sum_t[0] =  (i == 0 ? 0 : lm_sum[0][LSIZE_2 + LOG_LSIZE]);
        sqsum_t[0] =  (i == 0 ? (float4)0 : lm_sqsum[0][LSIZE_2 + LOG_LSIZE]);
        sum_t[1] =  (i == 0 ? 0 : lm_sum[1][LSIZE_2 + LOG_LSIZE]);
        sqsum_t[1] =  (i == 0 ? (float4)0 : lm_sqsum[1][LSIZE_2 + LOG_LSIZE]);
        barrier(CLK_LOCAL_MEM_FENCE);

        int bf_loc = lid + GET_CONFLICT_OFFSET(lid);
        lm_sum[0][bf_loc] = src_t[0];
        lm_sqsum[0][bf_loc] = sqsrc_t[0];

        lm_sum[1][bf_loc] = src_t[1];
        lm_sqsum[1][bf_loc] = sqsrc_t[1];

        int offset = 1;
        for(int d = LSIZE >> 1 ;  d > 0; d>>=1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ai = offset * (((lid & 127)<<1) +1) - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi]  +=  lm_sum[lid >> 7][ai];
                lm_sqsum[lid >> 7][bi]  +=  lm_sqsum[lid >> 7][ai];
            }
            offset <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < 2)
        {
            lm_sum[lid][LSIZE_2 + LOG_LSIZE] = 0;
            lm_sqsum[lid][LSIZE_2 + LOG_LSIZE] = 0;
        }
        for(int d = 1;  d < LSIZE; d <<= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset >>= 1;
            int ai = offset * (((lid & 127)<<1) +1) - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi] += lm_sum[lid >> 7][ai];
                lm_sum[lid >> 7][ai] = lm_sum[lid >> 7][bi] - lm_sum[lid >> 7][ai];

                lm_sqsum[lid >> 7][bi] += lm_sqsum[lid >> 7][ai];
                lm_sqsum[lid >> 7][ai] = lm_sqsum[lid >> 7][bi] - lm_sqsum[lid >> 7][ai];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
//        if(gid == 0 && (i + lid) <= rows)
 //       {
  //          sum[sum_offset + i + lid] = 0;
 //           sqsum[sqsum_offset + i + lid] = 0;
 //       }
  //      if(i + lid == 0)
//        {
   //         int loc0 = gid * 2 * sum_step;
  //          int loc1 = gid * 2 * sqsum_step;
   //         for(int k = 1; k <= 8; k++)
   //         {
   //             if(gid * 8 + k > cols) break;
    //            sum[sum_offset + loc0 + k * sum_step / 4] = 0;
   //             sqsum[sqsum_offset + loc1 + k * sqsum_step / 4] = 0;
     //       }
   //     }
   //copy from imgproc_integral_sum.cl
  //  int loc_s0 = sum_offset + gid * 2 * sum_step +                i + lid -1 , loc_s1 = loc_s0 + sum_step ;


       // int loc_s0 = sum_offset + gid * 2 * sum_step + sum_step / 4 + i + lid, loc_s1 = loc_s0 + sum_step ;
        //int loc_sq0 = sqsum_offset + gid * 2 * sqsum_step + sqsum_step / 4 + i + lid, loc_sq1 = loc_sq0 + sqsum_step ;
		int loc_s0 = sum_offset + gid * 2 * sum_step +              i + lid - 1 , loc_s1 = loc_s0 + sum_step ;
        int loc_sq0 = sqsum_offset + gid * 2 * sqsum_step +         i + lid - 1, loc_sq1 = loc_sq0 + sqsum_step ;
        if(lid > 0 && (i+lid) <= rows)
        {
            lm_sum[0][bf_loc] += sum_t[0];
            lm_sum[1][bf_loc] += sum_t[1];
            lm_sqsum[0][bf_loc] += sqsum_t[0];
            lm_sqsum[1][bf_loc] += sqsum_t[1];
            sum_p = (__local int*)(&(lm_sum[0][bf_loc]));
            sqsum_p = (__local float*)(&(lm_sqsum[0][bf_loc]));
            for(int k = 0; k < 4; k++)
            {
                if(gid * 8 + k >= cols) break;
                sum[loc_s0 + k * sum_step / 4] = sum_p[k];
                sqsum[loc_sq0 + k * sqsum_step / 4] = sqsum_p[k];
            }
            sum_p = (__local int*)(&(lm_sum[1][bf_loc]));
            sqsum_p = (__local float*)(&(lm_sqsum[1][bf_loc]));
            for(int k = 0; k < 4; k++)
            {
                if(gid * 8 + 4 + k >= cols) break;
                sum[loc_s1 + k * sum_step / 4] = sum_p[k];
                sqsum[loc_sq1 + k * sqsum_step / 4] = sqsum_p[k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
  		//	if(lid==0 && gid==0)
		//		printf("sum=%d,sqsum=%f\n",sum[lid],sqsum[lid]  );
}




