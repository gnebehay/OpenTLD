


#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void cltldOverlapRect(
    const unsigned int bb10,
    const unsigned int bb11,
    const unsigned int bb12,
    const unsigned int bb13,
	__global int* oclbuffWindowsOffset,
    __global float* overlap,
	const unsigned int TLD_WINDOW_SIZE
	)
{
	int windowIdx = get_global_id(0); 
	int min,max,sum1,sum2;
	__global int *bb2= oclbuffWindowsOffset + windowIdx * TLD_WINDOW_SIZE;
 
		if (bb10 > bb2[0] + bb2[2])
		{
			overlap[windowIdx] = 0.0;
			goto finished;
		}

		if (bb11 > bb2[1] + bb2[3])
		{
			overlap[windowIdx] = 0.0;
					goto finished;
		}

		if (bb10 + bb12 < bb2[0])
		{
			overlap[windowIdx] = 0.0;
			goto finished;
		}

		if (bb11 + bb13 < bb2[1])
		{
			overlap[windowIdx] = 0.0;
					goto finished;
		}
		else
		{
			sum1 =  bb10 + bb12;
			sum2 = bb2[0] + bb2[2];
			min = sum1<sum2? sum1:sum2;
			max = bb10>bb2[0]? bb10:bb2[0];
			int colInt =min-max;
			 
			 sum1 =  bb11 + bb13;
			sum2 = bb2[1] + bb2[3];
			min = sum1<sum2? sum1:sum2;
			max = bb11>bb2[1]? bb11:bb2[1];
			int rowInt =min-max;

			int intersection = colInt * rowInt;
			int area1 = bb12 * bb13;
			int area2 = bb2[2] * bb2[3];
			overlap[windowIdx] = intersection / (float)(area1 + area2 - intersection);
		}
		
	finished:
		;
		//if(overlap[windowIdx]!=0.0)
		//if( windowIdx==249404 )
		//{	
		//	for(int i=0;i<100;i++)
		//		printf(".....windowIdx =%d\n",windowIdx);
		//		printf("windowIdx=%d,overlap[xx]=%f\n",windowIdx,overlap[windowIdx]);	
 		//}
 }	 
	
 
 		