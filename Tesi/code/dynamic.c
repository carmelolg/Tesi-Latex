dim3 grid(3,2,2);
dim3 block(4,2);

__global__ void child(/* arguments */) {

	/* algorithm */

}

__global__ void kernel(/* arguments */) {

	child<<<grid, block>>>(/* arguments */);
}

int main() {

	kernel<<<grid, block>>>(/* arguments */);
}
