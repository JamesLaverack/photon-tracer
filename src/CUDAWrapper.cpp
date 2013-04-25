#include "CUDAWrapper.h"
#include <curand_kernel.h>
#include <stdio.h>
__global__ void setup_kernel(curandState *state, int iterations, int total_threads, int seed, int device_id, int i)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x)*iterations;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
	//for(int i=0;i<iterations;i++) {
		curand_init(seed, (id+i)+(device_id*total_threads), 0, &state[id+i]);
	//}
}

__global__ void img_setup_kernel(float4 *value, int y, int width)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x);
    /* Each thread gets same seed, a different sequence 
       number, no offset */
	//for(int i=0;i<iterations;i++) {
	value[width*y + id] = make_float4(0, 0, 0, 0);
	//}
}

__global__ void img_acc_kernel(float4 **values, float4 *accumulate, int y, int width, int height, int num_buffers)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x);
    /* Each thread gets same seed, a different sequence 
       number, no offset */
	accumulate[width*y + id] = make_float4(0, 0, 0, 0);
	for(int i=0;i<num_buffers;i++) {
		accumulate[width*y + id].x += values[i][width*y + id].x;
		accumulate[width*y + id].y += values[i][width*y + id].y;
		accumulate[width*y + id].z += values[i][width*y + id].z;
		accumulate[width*y + id].w += values[i][width*y + id].w;
	}
}

namespace photonCPU {

void CUDAWrapper::curand_setup(int num_blocks, int threads_per_block, int total_threads, void** states, int seed, int device_id) {
	int total_kernels = num_blocks*threads_per_block;
	int iterations = total_threads/total_kernels;
	printf("Init random states with:\n");
	printf("    %d blocks.\n", num_blocks);
	printf("    %d threads per block.\n", threads_per_block);
	printf("    %d iterations each.\n", iterations);
	printf("    %d total random generators.\n", total_threads);
	for(int i=0;i<iterations;i++) {
		setup_kernel<<<num_blocks, threads_per_block>>>((curandState*) *states, iterations, total_threads, seed, device_id, i);
	}
}

void CUDAWrapper::img_accumulate(void*** dev_ptrs, void** destination, int buffers, int width, int height) {
	int num_blocks= 10;
	int threads_per_block = height/num_blocks;
	for(int i=0;i<height;i++) {
		img_acc_kernel<<<num_blocks, threads_per_block>>>((float4**) *dev_ptrs, (float4*) *destination, i, width, height, buffers);
	}
}

void CUDAWrapper::img_setup(void** states, int width, int height) {
	int num_blocks= 10;
	int threads_per_block = height/num_blocks;
	for(int i=0;i<height;i++) {
		img_setup_kernel<<<num_blocks, threads_per_block>>>((float4*) *states, i, width);
	}
}

} /* photonCPU */