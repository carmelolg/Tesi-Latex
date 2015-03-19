#include <calCL2D.h>

int main(){

 CALOpenCL * calOpenCL = calclCreateCALOpenCL();
 //get all available platforms
 calclInitializePlatforms(calOpenCL);
 //get all available devices
 calclInitializeDevices(calOpenCL);

 //get the first device on the first platform
 CALCLdevice device = calclGetDevice(calOpenCL, 0, 0);

 //create a context
 CALCLcontext context = calclcreateContext(&device, 1);
 
 return 0;
}