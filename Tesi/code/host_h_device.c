//Using cudaMemcpy
cudaMemcpy(hybrid_model->i,host_model->i, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);

//Whole model passed between host and device
cudaMemcpy(hybrid_model,host_model, sizeof(CudaCALModel2D), cudaMemcpyHostToDevice);
