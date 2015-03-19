#include <CL/cl.h>

int main(){
 cl_platform_id * platforms;									
 cl_uint num_platforms;
 
 //get platforms number
 clGetPlatformIDs(1, NULL, &num_platforms);											
 platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
 //get platform
 clGetPlatformIDs(num_platforms, platforms, NULL);

 cl_platform_id first_platform = platforms[0];

 cl_device_id *devices;
 cl_uint num_devices;

 //get number of device in the first platform
 clGetDeviceIDs(first_platform, CL_DEVICE_TYPE_ALL,1, NULL, &num_devices);
 devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
 //get all devices from the first platform
 clGetDeviceIDs(first_platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

 cl_context context;

 //create a context
 context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

 return 0;
}