struct CudaCALModel2D* calCudaAllocatorModel(struct CudaCALModel2D *model){

	cudaMallocHost((void**)&copy_model, sizeof(struct CudaCALModel2D), cudaHostAllocPortable);

	memcpy(copy_model,model,sizeof(struct CudaCALModel2D));

	cudaMalloc((void**)&copy_model->i,model->sizeof_X*sizeof(int));
	cudaMalloc((void**)&copy_model->j,model->sizeof_X*sizeof(int));

	if(model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaMalloc((void**)&copy_model->activecell_flags,model->rows*model->columns*sizeof(CALbyte));
		cudaMalloc((void**)&copy_model->activecell_index,model->rows*model->columns*sizeof(CALint));
		cudaMalloc((void**)&copy_model->array_of_index_result, model->rows*model->columns*sizeof(CALint));
	}

	if(model->sizeof_pQb_array > 0){
		cudaMalloc((void**)&copy_model->pQb_array_current,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte));
		cudaMalloc((void**)&copy_model->pQb_array_next,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte));
	}
	if(model->sizeof_pQi_array > 0){
		cudaMalloc((void**)&copy_model->pQi_array_current,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint));
		cudaMalloc((void**)&copy_model->pQi_array_next,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint));
	}
	if(model->sizeof_pQr_array > 0){
		cudaMalloc((void**)&copy_model->pQr_array_current,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal));
		cudaMalloc((void**)&copy_model->pQr_array_next,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal));
	}

	return copy_model;
}
