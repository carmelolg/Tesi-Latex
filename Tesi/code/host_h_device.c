//Using cudaMemcpy
cudaMemcpy(ibrid_model->i,host_model->i, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);

//Whole model passed between host and device
cudaMemcpy(ibrid_model,host_model, sizeof(CudaCALModel2D), cudaMemcpyHostToDevice);
