/*
 * Renderer.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef CUDAWRAPPER_H_
#define CUDAWRAPPER_H_

#include <cuda.h>
#include <ctime>
#include <sys/time.h>
namespace photonCPU {

class CUDAWrapper {
public:
	void curand_setup(int num_rngs, void** states, int seed, int device_id);
	void img_setup(void** states, int width, int height);
	void img_accumulate(void*** dev_ptrs, void** destination, int buffers, int width, int height);
};

} /* namespace photonCPU */
#endif /* RENDERER_H_ */
