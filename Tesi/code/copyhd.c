CALbyte calInitializeInGPU2D(struct CudaCALModel2D* model, struct CudaCALModel2D *d_model){

	CALbyte result = CAL_TRUE;

	calCudaAllocatorModel(model);

	cudaMemcpy(copy_model->i,model->i, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);
	cudaMemcpy(copy_model->j,model->j, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);

	if(model->OPTIMIZATION == CAL_OPT_ACTIVE_CELLS){
		cudaMemcpy(copy_model->activecell_flags,model->activecell_flags, sizeof(CALbyte)*model->rows*model->columns, cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->activecell_index,model->activecell_index, sizeof(CALint)*model->rows*model->columns, cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->array_of_index_result,model->array_of_index_result, sizeof(CALint)*model->rows*model->columns, cudaMemcpyHostToDevice);
	}

	if(model->sizeof_pQb_array > 0){
		cudaMemcpy(copy_model->pQb_array_current,model->pQb_array_current, model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->pQb_array_next,model->pQb_array_next, model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyHostToDevice);
	}
	if(model->sizeof_pQi_array > 0){
		cudaMemcpy(copy_model->pQi_array_current,model->pQi_array_current, model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->pQi_array_next,model->pQi_array_next, model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyHostToDevice);
	}
	if(model->sizeof_pQr_array > 0){
		cudaMemcpy(copy_model->pQr_array_current,model->pQr_array_current, model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyHostToDevice);
		cudaMemcpy(copy_model->pQr_array_next,model->pQr_array_next, model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_model, copy_model, sizeof(struct CudaCALModel2D), cudaMemcpyHostToDevice);

	return result;
}
