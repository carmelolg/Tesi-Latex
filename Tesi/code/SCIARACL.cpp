#include "io.h"
#include "Sciara.h"
#include "calCL2D.h"

#define KER_SCIARA_ELEMENTARY_PROCESS_ONE "updateVentsEmission"
#define KER_SCIARA_ELEMENTARY_PROCESS_TWO "empiricalFlows"
#define KER_SCIARA_ELEMENTARY_PROCESS_THREE "width_update"
#define KER_SCIARA_ELEMENTARY_PROCESS_FOUR "updateTemperature"
#define KER_SCIARA_ELEMENTARY_PROCESS_FIVE  "removeActiveCells"
#define KER_SCIARA_STOP_CONDITION "stopCondition"
#define KER_SCIARA_STEERING  "steering"

int main(int argc, char** argv) {

 /* ... some initializations ...*/

 //Load OpenCL platforms and devices
 CALOpenCL * calOpenCL = calclCreateCALOpenCL();
 calclInitializePlatforms(calOpenCL);
 calclInitializeDevices(calOpenCL);
 CALCLdevice device = calclGetDevice(calOpenCL, platformNum, deviceNum);

 //Context definition
 CALCLcontext context = calclcreateContext(&device, 1);

 //Load lib and user kernels
 CALCLprogram program = calclLoadProgramLib2D(context, device, kernelSrc, kernelInc);

 /*... Load SCIARA conf from files ...*/
 
 CALCLToolkit2D * sciaraToolkit = calclCreateToolkit2D(sciara->model, context, program, device, CALCL_NO_OPT);
 //Load kernels from compiled program
 CALCLkernel kernel_elementary_process_one = calclGetKernelFromProgram(&program, KER_SCIARA_ELEMENTARY_PROCESS_ONE);
 CALCLkernel kernel_elementary_process_two = calclGetKernelFromProgram(&program, KER_SCIARA_ELEMENTARY_PROCESS_TWO);
 CALCLkernel kernel_elementary_process_three = calclGetKernelFromProgram(&program, KER_SCIARA_ELEMENTARY_PROCESS_THREE);
 CALCLkernel kernel_elementary_process_four = calclGetKernelFromProgram(&program, KER_SCIARA_ELEMENTARY_PROCESS_FOUR);
 CALCLkernel kernel_stop_condition = calclGetKernelFromProgram(&program, KER_SCIARA_STOP_CONDITION);
 CALCLkernel kernel_steering = calclGetKernelFromProgram(&program, KER_SCIARA_STEERING);


 //Set some additional kernel args
 
 ventsMapper(sciaraToolkit, context, kernel_elementary_process_one);
 CALCLmem elapsed_timeBuffer = calclCreateBuffer(context, &sciara->elapsed_time, sizeof(CALreal));
 clSetKernelArg(kernel_elementary_process_one, MODEL_ARGS_NUM + 2, sizeof(CALCLmem), &elapsed_timeBuffer);
 clSetKernelArg(kernel_elementary_process_one, MODEL_ARGS_NUM + 5, sizeof(Parameters), &sciara->parameters);

 clSetKernelArg(kernel_elementary_process_two, MODEL_ARGS_NUM, sizeof(Parameters), &sciara->parameters);

 CALCLmem mbBuffer = calclCreateBuffer(context, sciara->substates->Mb->current, sizeof(CALbyte) * (sciara->rows * sciara->cols));
 CALCLmem mslBuffer = calclCreateBuffer(context, sciara->substates->Msl->current, sizeof(CALreal) * (sciara->rows * sciara->cols));
 clSetKernelArg(kernel_elementary_process_four, MODEL_ARGS_NUM, sizeof(CALCLmem), &mbBuffer);
 clSetKernelArg(kernel_elementary_process_four, MODEL_ARGS_NUM + 1, sizeof(CALCLmem), &mslBuffer);
 clSetKernelArg(kernel_elementary_process_four, MODEL_ARGS_NUM + 2, sizeof(Parameters), &sciara->parameters);

 //adding elementary processes
 calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_one);
 calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_two);
 calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_three);
 calclAddElementaryProcessKernel2D(sciaraToolkit, sciara->model, &kernel_elementary_process_four);

 //set steering kernel
 clSetKernelArg(kernel_steering, MODEL_ARGS_NUM, sizeof(CALCLmem), &mbBuffer);
 clSetKernelArg(kernel_steering, MODEL_ARGS_NUM + 1, sizeof(Parameters), &sciara->parameters);
 clSetKernelArg(kernel_steering, MODEL_ARGS_NUM + 2, sizeof(CALCLmem), &elapsed_timeBuffer);
 calclSetSteeringKernel2D(sciaraToolkit, sciara->model, &kernel_steering);

 //set stop condition kernel
 clSetKernelArg(kernel_stop_condition, MODEL_ARGS_NUM, sizeof(Parameters), &sciara->parameters);
 clSetKernelArg(kernel_stop_condition, MODEL_ARGS_NUM + 1, sizeof(CALCLmem), &elapsed_timeBuffer);
 calclSetStopConditionKernel2D(sciaraToolkit, sciara->model, &kernel_stop_condition);

 //run the simulation
 calclRun2D(sciaraToolkit, sciara->model, steps);

 /*... save computation results and deallocate resources */
   
 return 0;

}
