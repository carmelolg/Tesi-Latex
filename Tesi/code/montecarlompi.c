#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[]) {

	const char Usage[] = "Usage: pi <steps> <repeats> (try 1000000 4)";
	if (argc < 3) {
		printf("%s \n", Usage);
		return (1);
	}
	int num_steps = atoi(argv[1]);
	int num_repeats = atoi(argv[2]);
	int my_rank, p, range, y, low, high, i, j, source, count = 0;
	int local_count = 0;
	int root = 0, tag = 0;
	double start, end, total;
	MPI_Status status; /*	      return status for receive	*/

	//A little throwaway parallel section just to show num threads
	/* Start up MPI */
	MPI_Init(&argc, &argv);

	/* Find out process rank  */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* Find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if (my_rank == 0) {
		printf(
				"Computing pi via Monte Carlo using %d steps, repeating %d time \n",
				num_steps, num_repeats);
	}

	start = omp_get_wtime();
	if (p <= num_repeats) {
		range = num_repeats / p;
		low = my_rank * range;
		high = low + range;
	} else {
		low = 0;
		high = num_repeats;
	}
	for (i = low; i < high; i++) {
		count = 0;
		local_count = 0;
		for (j = 0; j < num_steps; j++) {
			double x = (double) rand() / RAND_MAX;
			double y = (double) rand() / RAND_MAX;
			if (x * x + y * y <= 1)
				local_count++;
		}

		printf("Rank %d Local count %d \n", my_rank, local_count);

		if (my_rank == 0) {
			count = local_count;
			for (source = 1; source < p; source++) {
				MPI_Recv(&local_count, 1, MPI_INT, source, tag, MPI_COMM_WORLD,
						&status);
				count = count + local_count;
			}

		} else {
			MPI_Send(&local_count, 1, MPI_INT, root, tag, MPI_COMM_WORLD);
		}

	}
	if (my_rank == 0) {
		double pi = (double) count / (num_steps * p) * 4;
		printf("pi = %g\n", pi);
		end = omp_get_wtime();
		total = end - start;
		printf("Time: %1.24f seconds \n", total);
	}
	MPI_Finalize();
	exit(0);
}
