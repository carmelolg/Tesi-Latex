#include <CL/cl.h>

#define NUM_FILES 2
#define PROGRAM_FILE_1 "P1.cl"
#define PROGRAM_FILE_2 "P2.cl"
#define KERNEL_NAME "k1"

int main(){

 /*
 .... allocate and initialize some variables ....
 */

 cl_program program;
 FILE *program_handle;
 char *program_buffer[NUM_FILES];
 const char *file_name[] = {PROGRAM_FILE_1, PROGRAM_FILE_2};
 size_t program_size[NUM_FILES];

 //read all kernels functions from source files
 for(i=0; i<NUM_FILES; i++){
 	program_handle = fopen(file_name[i], "r");
 	fseek(program_handle, 0, SEEK_END);
 	program_size[i] = ftell(program_handle);
 	rewind(program_handle);
 	program_buffer[i] = (char*)malloc(program_size[i]+1);
 	program_buffer[i][program_size[i]] = '\0';
 	fread(program_buffer[i], sizeof(char),
 	program_size[i], program_handle);
 	fclose(program_handle);
 }

 //create a program
 program = clCreateProgramWithSource(context, NUM_FILES,(const char**)program_buffer, program_size, &err);
 //build a program
 clBuildProgram(program, 1, &device ,NULL, NULL, NULL);
 //create a kernel
 cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, NULL);

 return 0;
}