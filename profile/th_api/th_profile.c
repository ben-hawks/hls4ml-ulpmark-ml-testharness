/**
 * Copyright (C) EEMBC(R). All Rights Reserved
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 */
//"profile/th_api/th_profile.h"
#include "th_profile.h"

/* From profile/ee_buffer.c */
extern uint8_t *gp_buff;
extern size_t   g_buff_size;
extern size_t   g_buff_pos;
/* from ../../../main.c */
extern uint32_t src_mem_size;
extern uint32_t dst_mem_size;
uint8_t floatsize = sizeof(float);

/**
 * Copy the octet buffer `gp_buff` into your input tensor in the proper
 * format.
 */
void
th_load_tensor(void)
{
    Xil_DCacheFlushRange((UINTPTR)src_mem, src_FEATURE_COUNT * sizeof(unsigned char));
    Xil_DCacheFlushRange((UINTPTR)dst_mem, dst_FEATURE_COUNT * sizeof(unsigned char));

	XAnomaly_detector_axi_Set_in_V(&do_anomaly_detector, (unsigned)src_mem);
	XAnomaly_detector_axi_Set_out_V(&do_anomaly_detector, (unsigned)dst_mem);

	float data_flt = 0.0;
	int skip = 4; // take only every nth frame, where n = skip
	slices = 4;
	bins = 128;
	int newcnt = 0;
	printf("g_buff_size: %i ",g_buff_size);
	printf("src_mem_size: %i\r\n",src_mem_size);
    for (int i = 0; i < slices*bins; i+=skip) { //Load input data, number of features = slices/bins (4x128 in our case), only loading every nth
    	//memcpy(&data_flt,&gp_buff[i*floatsize],(int)floatsize);
    	//unsigned char data_fxd = (unsigned char)data_flt;
    	src_mem[i] = gp_buff[i*floatsize];
    	printf("Load Iteration with data %i: Feature # %i\r\n",(unsigned int)src_mem[i],(i/skip)+1);
    	newcnt++;
    }
    for (int i = 0; i < dst_mem_size; i++) {//Init DST mem with 0's
        dst_mem[i] = 0x0;
    }
    printf("Loaded features: %i\r\n",newcnt);
    //malloc_stats();
}

/**
 * Perform a single inference.
 */
void
th_infer(void)
{
	//Normally required to set pointers for input/output of the detector, but we're looping over one sample here, so its not
	//XAnomaly_detector_axi_Set_in_V(&do_anomaly_detector, (unsigned)src_mem);
	//XAnomaly_detector_axi_Set_out_V(&do_anomaly_detector, (unsigned)dst_mem);

	//start accelerator
	XAnomaly_detector_axi_Start(&do_anomaly_detector);
	/* polling for accelerator to finish*/
	while (!XAnomaly_detector_axi_IsDone(&do_anomaly_detector));
}

void
th_results(void)
{
    float *results  = NULL;
    size_t nresults = 0;
    /* USER CODE 1 BEGIN */
    float result = 0;
    float sum = 0.0;
    /* Populate results[] and n from the fp32 prediction tensor. */
    printf("INFO: Starting results iteration, src_FEATURE_COUNT %i * floatsize %i\r\n",src_FEATURE_COUNT,(int)floatsize);
    for(size_t i = 0; i < src_FEATURE_COUNT*floatsize; i+=floatsize){ //find the error score of each feature, then average over all features
    	//printf("INFO: Iteration %i\r\n",(i/floatsize));
    	//printf("INFO: SRC Mem:  %i",(src_mem[i]));
    	//printf(" DST Mem:  %i\r\n",(dst_mem[i]));
    	uint8_t diff = src_mem[i]-dst_mem[i];
    	float sq = (float)diff*(float)diff;
    	sum += sq;
    	printf("INFO: Anomaly Feature # %i diff, sum: %i, %.3f\r\n",(i/floatsize),diff,sum);
    }
    float mse = sum/(src_FEATURE_COUNT);
    printf("INFO: Anomaly Score (MSE): %.3f\r\n",mse);
    result = mse;

//#warning "th_results() not implemented"
    /* USER CODE 1 END */
    /**
     * The results need to be printed back in exactly this forth_printf("m-results-[%0.3f]\r\n", result);mat; if easier
     * to just modify this loop than copy to results[] above, do that.
     */
    th_printf("m-results-[%0.3f]\r\n", result);
}
