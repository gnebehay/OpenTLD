#define opencl 0
#if   opencl
#define CPU 0
#else 
#define CPU 1
#endif

#define clDetect 0
#define clIntegral 0
#define clIntegral_m 0
#define clIntegral_m_1 0
#define clOverlap 0
#define clNNClassifier  0

#define PrintTime_overlap 0
#define PrintTime_detect 0
#define PrintTime_learn 0
#define PrintTime_integral 0
#define PrintTime_fuseHypotheses 0
#define SELECTED_DEVICE_ID 0