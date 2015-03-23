//Allocazione sul device
cudaMalloc((void**)&data, sizeof(...));

//Trasferimento dei dati da CPU a GPU
cudaMemcpy(void *dst, void *src, sizeof(...), cudaMemcpyHostToDevice);
//Trasferimento dei dati da GPU a CPU
cudaMemcpy(void *dst, void *src, sizeof(...), cudaMemcpyDeviceToHost);
