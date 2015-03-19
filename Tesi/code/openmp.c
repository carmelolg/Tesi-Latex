#include <omp.h>

float num_subintervals = 10000; float subinterval;
#define NUM_THREADS 5

void main ()
{
	int i; float x, pi, area = 0.0;
	subinterval = 1.0 / num_subintervls;

	omp_set_num_threads (NUM_THREADS)
	#pragma omp parallel for reduction(+:area) private(x)
	for (i=1; i<= num_subintervals; i++) {
		x = (i-0.5)*subinterval;
		area = area + 4.0 / (1.0+x*x);
	}
	pi = subinterval * area;
}