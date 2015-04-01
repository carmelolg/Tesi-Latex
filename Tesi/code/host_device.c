//Using cudaMemcpy
cudaMemcpy(device_model->i,host_model->i, sizeof(CALint)*model->sizeof_X, cudaMemcpyHostToDevice);
