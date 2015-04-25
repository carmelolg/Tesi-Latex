int main()
{
	time_t start_time, end_time;

	cudaProfilerStart();

	//Model and simulation definitions
	struct CudaCALModel2D* sciara_fv2;
	struct CudaCALRun2D* simulation_sciara_fv2;

#ifdef ACTIVE_CELLS
	sciara_fv2 = calCudaCADef2D (rows, cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else  
	sciara_fv2 = calCudaCADef2D (rows, cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	//Model allocated on device
	struct CudaCALModel2D* device_sciara_fv2 = calCudaAlloc();

	//Add transition function's elementary processes
	calCudaAddElementaryProcess2D(sciara_fv2, updateVentsEmission);
	calCudaAddElementaryProcess2D(sciara_fv2, empiricalFlows);
	calCudaAddElementaryProcess2D(sciara_fv2, width_update);
	calCudaAddElementaryProcess2D(sciara_fv2, updateTemperature);

#ifdef ACTIVE_CELLS
	calCudaAddElementaryProcess2D(sciara_fv2, removeActiveCells);
#endif

	//Add substates
	calCudaAddSubstate2Dr(sciara_fv2,NUMBER_OF_SUBSTATES_REAL);
	calCudaAddSubstate2Di(sciara_fv2,NUMBER_OF_SUBSTATES_INT);
	calCudaAddSubstate2Db(sciara_fv2,NUMBER_OF_SUBSTATES_BYTE);

	//Load configuration
	calCudaLoadSubstate2Dr(sciara_fv2, DEM_PATH, ALTITUDE);
	calCudaLoadSubstate2Di(sciara_fv2, VENTS_PATH, VENTS);
	
	//calCudaLoadSubstate2Dr(sciara_fv2, THICKNESS_PATH, THICKNESS);
	calCudaLoadSubstate2Dr(sciara_fv2, TEMPERATURE_PATH, TEMPERATURE);
	calCudaLoadSubstate2Dr(sciara_fv2, SOLIDIFIED_LAVA_THICKNESS_PATH, SOLIDIFIED);

	//Copy data from CPU to GPU
	calInitializeInGPU2D(sciara_fv2,device_sciara_fv2);

	//Check errors otherwise print the message in input
	cudaErrorCheck("Data initialized on device\n");

	simulation_sciara_fv2 = calCudaRunDef2D(device_sciara_fv2, sciara_fv2, 1, STEPS, CAL_UPDATE_IMPLICIT);

	//Add init, steering and stop condition
	calCudaRunAddInitFunc2D(simulation_sciara_fv2, simulationInitialize);
	calCudaRunAddSteeringFunc2D(simulation_sciara_fv2, steering);
	calCudaRunAddStopConditionFunc2D(simulation_sciara_fv2, stopCondition);

	//Start simulation
	printf ("Starting simulation...\n");
	start_time = time(NULL);
	calCudaRun2D(simulation_sciara_fv2, grid, block);

	//Send data to CPU
	calSendDataGPUtoCPU(sciara_fv2,device_sciara_fv2);

	cudaErrorCheck("Final configuration sent to CPU\n");
	end_time = time(NULL);
	printf ("Simulation terminated.\nElapsed time: %d\n", end_time-start_time);

	//Saving configuration
	calCudaSaveSubstate2Dr(sciara_fv2, O_DEM_PATH, ALTITUDE);
	calCudaSaveSubstate2Dr(sciara_fv2, O_SOLIDIFIED_LAVA_THICKNESS_PATH, SOLIDIFIED);
	calCudaSaveSubstate2Dr(sciara_fv2, O_TEMPERATURE_PATH, TEMPERATURE);
	calCudaSaveSubstate2Dr(sciara_fv2, O_THICKNESS_PATH, THICKNESS);
	calCudaSaveSubstate2Di(sciara_fv2, O_VENTS_PATH, VENTS);

	//Check errors 
	cudaErrorCheck("Data saved on output file\n");

	//Finalizations
	calCudaRunFinalize2D(simulation_sciara_fv2);
	calCudaFinalize2D(sciara_fv2, device_sciara_fv2);
	cudaProfilerStop();

	system("pause");
	return 0;
}