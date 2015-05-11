#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv)
{
	int myid, numprocs;

	MPI_Init(&argc, &argv); // This is mandatory at the start of the parallel part of program

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	/* Print out my rank and this run's PE size */
	printf("Hello World from %d\n",myid);
	printf("The number of procs is %d\n",numprocs);

	MPI_Finalize(); // This is mandatory at the end of the parallel part of the program

}
