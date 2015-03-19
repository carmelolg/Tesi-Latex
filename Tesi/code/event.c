#include <CL/cl.h>

void CL_CALLBACK process(cl_event event, cl_int status, void *data) {
printf("%s\n", (char*)data);
}


int main() {
 /*...*/
 cl_event ev;
 char[] msg = "Hello world!";
 //set callback function
 cl_int clSetEventCallback(ev, CL_COMPLETE, &process, (void*)msg);
 /*...*/
}