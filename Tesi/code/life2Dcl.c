#include <cal2D.h>
#include <cal2DIO.h>
#include <calCL2D.h>

#define ROWS 300
#define COLS 300
#define STEPS 1000

#define KERNEL_SRC "/kernel/"
#define KERNEL_NAME "lifeTransitionFunction"
#define SAVE_PATH"/data/lifeSubstateResult"

struct CALSubstate2Di * lifeSubstate;

void init(struct CALModel2D* life){
 calInit2Di(life, lifeSubstate, 0, 2, 1);
 calInit2Di(life, lifeSubstate, 1, 0, 1);
 calInit2Di(life, lifeSubstate, 1, 2, 1);
 calInit2Di(life, lifeSubstate, 2, 1, 1);
 calInit2Di(life, lifeSubstate, 2, 2, 1);
}

int main() {

 //life model definition
 struct CALModel2D * model = calCADef2D(ROWS, COLS, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
 lifeSubstate = calAddSubstate2Di(model);
 calInitSubstate2Di(model,lifeSubstate,0);

 CALOpenCL * calOpenCL = calclCreateCALOpenCL();
 //get all available platforms
 calclInitializePlatforms(calOpenCL);
 //get all available devices
 calclInitializeDevices(calOpenCL);
 //get the first device on the first platform
 CALCLdevice device = calclGetDevice(calOpenCL, 0, 0);
 
 //create a context
 CALCLcontext context = calclcreateContext(&device, 1);
  //create a program compiling all kernel source files at the path KERNEL_SRC
 CALCLprogram program = calclLoadProgramLib2D(context, device,KERNEL_SRC, NULL);
 //create a toolkit containing all buffer object	
 CALCLToolkit2D * toolkit = calclCreateToolkit2D(model, context, program,device, CALCL_NO_OPT);

 //get life transition function
 cl_kernel elementaryProcess = calclGetKernelFromProgram(&program, KERNEL_NAME);
 //add life transition function to elementary processes list
 calclAddElementaryProcessKernel2D(toolkit,parallelModel,&elementaryProcess);
 //execute a simulation
 calclRun2D(toolkit, parallelModel, STEPS);
 //write on file simulation result
 calSaveSubstate2Di(parallelModel, lifeSubstateParallel, SAVE_PATH);

 //deallocate allocated resources
 calclFinalizeCALOpencl(calOpenCL);
 calclFinalizeToolkit2D(toolkit);
 calFinalize2D(model);

 return 0;
}
