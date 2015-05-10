#include ".\include\cal2D.cuh"
#include ".\include\cal2DIO.cuh"
#include ".\include\cal2DToolkit.cuh"
#include ".\include\cal2DRun.cuh"
#include <stdlib.h>
#include <time.h>

#include <iostream>
using namespace std;

#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//------------------------------------------------------------
//   THE GOL (Toy model) CELLULAR AUTOMATON
//------------------------------------------------------------

#define ROWS 1000
#define COLS 1000

#define STEPS 200
#define STEPS_THRESHOLD 200
#define CONFIGURATION_PATH "./data/map_1000x1000.txt"
#define OUTPUT_PATH "./data/final.txt"

#define NUMBER_OF_SUBSTATE_BYTE 1
#define NUMBER_OF_NEIGHBORHOOD 9

enum SUBSTATE_NAME{
	ALIVE = 0
};

struct CudaCALRun2D* gol_simulation;

CALint N = 25;
CALint M = 5;
dim3 block(N,M);
dim3 grid(COLS/block.x, ROWS/block.y);

__global__ void gol_computation(struct CudaCALModel2D* gol)
{
	CALint sum = 0, n, offset = calCudaGetIndex(gol);


	CALint i = calCudaGetIndexRow(gol, offset),j = calCudaGetIndexColumn(gol, offset);

	CALbyte myState = calCudaGet2Db(gol,offset, ALIVE);

	for(n=1; n<NUMBER_OF_NEIGHBORHOOD; n++){
		sum += calCudaGetX2Db(gol,offset,n, ALIVE);
	}

	if(myState == CAL_TRUE){
		if(sum != 2 && sum != 3){
			calCudaSet2Db(gol, offset, CAL_FALSE, ALIVE);
		}
	}else{
		if(sum == 3){
			calCudaSet2Db(gol, offset, CAL_TRUE, ALIVE);
		}
	}
}

__global__ void gol_simulation_init(struct CudaCALModel2D* gol)
{
	CALint offset = calCudaGetIndex(gol);

	//initializing substates to 0
	calCudaInit2Db(gol,offset,CAL_FALSE, ALIVE);

	// Glider 1000x1000
	if(offset == 1002 || offset == 2003 || offset == 3001 || offset == 3002 || offset == 3003){
		calCudaInit2Db(gol,offset,CAL_TRUE, ALIVE);
	}

}


__global__ void gol_simulation_stop(struct CudaCALModel2D* gol)
{
	CALint offset = calCudaGetIndex(gol);
	CALint i = calCudaGetIndexRow(gol, offset), j = calCudaGetIndexColumn(gol, offset);

	if(i == 14 && j == 9 && calCudaGet2Db(gol,offset,0))
		calCudaStop(gol);

}

int main()
{
	time_t start_time, end_time;
	cudaProfilerStart();

	//cadef
	struct CudaCALModel2D* gol = calCudaCADef2D (ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
	struct CudaCALModel2D* device_gol = calCudaAlloc();

	//rundef
	gol_simulation = calCudaRunDef2D(device_gol, gol, 1, CAL_RUN_LOOP, CAL_UPDATE_IMPLICIT);

	//add transition function's elementary processes
	calCudaAddElementaryProcess2D(gol, gol_computation);

	printf ("Starting alloc...\n");
	start_time = time(NULL);

	//add substates
	calCudaAddSubstate2Db(gol, NUMBER_OF_SUBSTATE_BYTE);

	//load configuration
	calCudaLoadSubstate2Db(gol, CONFIGURATION_PATH, ALIVE);

	//send data to GPU
	calInitializeInGPU2D(gol, device_gol);

	end_time = time(NULL);
	printf ("Alloc terminated.\nElapsed time: %d\n\n", end_time-start_time);

	cudaErrorCheck("Data initialized on device\n");

	//simulation configuration
	calCudaRunAddInitFunc2D(gol_simulation, gol_simulation_init);
	calCudaRunAddStopConditionFunc2D(gol_simulation,gol_simulation_stop);

	printf ("Starting simulation...\n");
	start_time = time(NULL);

	//simulation run
	calCudaRun2D(gol_simulation,grid,block);

	//send data to CPU
	calSendDataGPUtoCPU(gol,device_gol);

	cudaErrorCheck("Final configuration sent to CPU\n");

	end_time = time(NULL);
	printf ("Simulation terminated.\nElapsed time: %d\n\n", end_time-start_time);

	cudaProfilerStop();

	//saving configuration
	calCudaSaveSubstate2Db(gol, OUTPUT_PATH, ALIVE);

	cudaErrorCheck("Data saved on output file\n");

	//finalizations
	calCudaRunFinalize2D(gol_simulation);
	calCudaFinalize2D(gol, device_gol);

	system("pause");
	return 0;
}
