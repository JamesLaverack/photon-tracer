/*
 * Renderer.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef RENDERER_H_
#define RENDERER_H_

#include "Scene.h"
#include "PlaneObject.h"
#include "CameraMaterial.h"
#include "Image.h"

namespace photonCPU {

class Renderer {
private:
	CameraMaterial* mCameraMat;
	RenderObject* mCameraObject;
	Scene* mScene;
public:
	Renderer(Scene* pScene, int width, int height);
	virtual ~Renderer();
	void performRender(int photons, int argc_mpi, char* argv_mpi[]);
	void doRenderPass(int photons);
};

} /* namespace photonCPU */
#endif /* RENDERER_H_ */
