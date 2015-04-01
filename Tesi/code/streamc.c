struct Predicate
{
	__host__ __device__
		bool operator()(unsigned int x) const
	{
		return (x != -1);
	}
};

/* ... */

if (simulation->ca2D->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){

		CALint SIZE = (simulation->ca2D->rows*simulation->ca2D->columns);

		if(simulation->ca2D->activecell_size_current != simulation->h_device_ca2D->activecell_size_next){

			generateSetOfIndex<<<grid,block>>>(simulation->device_ca2D);

			cudaMemcpy(simulation->h_device_ca2D, simulation->device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);

			pp::compact(
				/* Input start pointer */
				simulation->h_device_ca2D->activecell_index,

				/* Input end pointer */
				simulation->h_device_ca2D->activecell_index+SIZE,

				/* Output start pointer */
				simulation->h_device_ca2D->array_of_index_result,

				/* Storage for valid element count */
				simulation->device_array_of_index_dim,

				/* Predicate */
				Predicate()
				);

			cudaMemcpy(simulation->device_ca2D, simulation->h_device_ca2D, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);

		}


		// resize of grid and blocks.
		simulation->ca2D->activecell_size_current = simulation->h_device_ca2D->activecell_size_next;
		CALint num_blocks = simulation->ca2D->activecell_size_current / ((block.x) * (block.y));
		grid.x = (num_blocks+2);
		grid.y = 1;

		//elementary process run
		elementary_process<<<grid,block>>>(simulation->device_ca2D);
	}
