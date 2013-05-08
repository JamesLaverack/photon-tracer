/*
 * Renderer.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef OPTIXRENDERER_H_
#define OPTIXRENDERER_H_

#include "Scene.h"
#include "PlaneObject.h"
#include "CameraMaterial.h"
#include "Image.h"
#include <iterator>
#include "CUDAWrapper.h"
#include <optixu/optixpp_namespace.h>
#ifdef PHOTON_MPI
	#include "mpi.h"
#endif /* MPI */

//#include <driver_types.h>
//using cuda::cudaError_t;

	using namespace optix;
	#include <curand_kernel.h>
	#include <cuda.h>
	#include <cuda_runtime_api.h>

namespace photonCPU {

class OptiXRenderer {
private:
	CameraMaterial* mCameraMat;
	optix::Material mCameraMatOptiX;
	RenderObject* mCameraObject;
	Scene* mScene;
	void convertToOptiXScene(optix::Context context, int width, int height, float film_location);
	void saveToPPMFile(char* filename, optix::float4* image, int width, int height);
	int index(int x, int y, int width);
	int toColourInt(float f, int maxVal);
public:
	OptiXRenderer(Scene* pScene);
	virtual ~OptiXRenderer();
	void performRender(long long int photons, int argc_mpi, char* argv_mpi[], int width, int height, float film_location);
	void doRenderPass(long long int photons);
};

} /* namespace photonCPU */
#endif /* RENDERER_H_ */
