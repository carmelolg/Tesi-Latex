/* ... */

CALint N = 16;
CALint M = 61;
dim3 block(N,M);
dim3 grid(COLS/block.x, ROWS/block.y);

/* ... */


//Start simulation in OpenCAL
calRun2D(simulation);

//Start simulation in OpenCAL-CUDA
calCudaRun2D(simulation, grid, block);

/* ... */
