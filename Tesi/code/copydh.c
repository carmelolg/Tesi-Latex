CALbyte calSendDataGPUtoCPU(struct CudaCALModel2D* model, struct CudaCALModel2D *d_model){

	CALbyte result = CAL_TRUE;

	cudaMemcpy(copy_model, d_model, sizeof(struct CudaCALModel2D), cudaMemcpyDeviceToHost);

	if(model->sizeof_pQb_array > 0){
		cudaMemcpy(model->pQb_array_current,copy_model->pQb_array_current,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyDeviceToHost);
		cudaMemcpy(model->pQb_array_next,copy_model->pQb_array_next,model->sizeof_pQb_array*model->rows*model->columns*sizeof(CALbyte), cudaMemcpyDeviceToHost);
	}
	if(model->sizeof_pQi_array > 0){
		cudaMemcpy(model->pQi_array_current,copy_model->pQi_array_current,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyDeviceToHost);
		cudaMemcpy(model->pQi_array_next,copy_model->pQi_array_next,model->sizeof_pQi_array*model->rows*model->columns*sizeof(CALint), cudaMemcpyDeviceToHost);
	}
	if(model->sizeof_pQr_array > 0){
		cudaMemcpy(model->pQr_array_current,copy_model->pQr_array_current,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyDeviceToHost);
		cudaMemcpy(model->pQr_array_next,copy_model->pQr_array_next,model->sizeof_pQr_array*model->rows*model->columns*sizeof(CALreal), cudaMemcpyDeviceToHost);
	}

	calCudaFinalizeModel();

	return result;
}
