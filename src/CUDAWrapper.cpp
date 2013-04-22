#include "CUDAWrapper.h"
#include <curand_kernel.h>
#include <stdio.h>
__global__ void setup_kernel(curandState *state, int iterations)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x)*iterations;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
	//printf("Test from thread %d\n", id);
	for(int i=0;i<iterations;i++) {
		curand_init(1232, id+i, 0, &state[id+i]);
	}
}

namespace photonCPU {

void CUDAWrapper::curand_setup(int num_blocks, int threads_per_block, int total_threads, void** states) {
	int total_kernels = num_blocks*threads_per_block;
	int iterations = total_threads/total_kernels;
	printf("Setup\n");
	printf("   %d blocks.", num_blocks);
	printf("   %d threads per block.", threads_per_block);
	printf("   %d iterations each.", iterations);
	printf("   %d total random generators.", total_threads);
	setup_kernel<<<num_blocks, threads_per_block>>>((curandState*) *states, iterations);
}

} /* photonCPU */