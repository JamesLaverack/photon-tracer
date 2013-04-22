/*
 * Renderer.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef CUDAWRAPPER_H_
#define CUDAWRAPPER_H_

#include <cuda.h>
namespace photonCPU {

class CUDAWrapper {
public:
	void curand_setup(int num_blocks, int threads_per_block, int total_threads, void** states);
};

} /* namespace photonCPU */
#endif /* RENDERER_H_ */
