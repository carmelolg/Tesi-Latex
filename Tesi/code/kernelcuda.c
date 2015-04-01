__global__ void elementary_process(struct CudaCALModel2D* model)
{
	CALreal value;
	CALint n, offset = calCudaGetOffset();

	value = calCudaGet2Dr(model, offset, SUBSTATE1_INDEX);

	value += calCudaGetX2Dr(model, offset, n, SUBSTATE2_INDEX)
		- calCudaGet2Dr(model, offset, SUBSTATE3_INDEX);

	calCudaSet2Dr(model, offset, value, SUBSTATE1_INDEX);
}
