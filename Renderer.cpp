/*
 * Renderer.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "Renderer.h"

namespace photonCPU {

Renderer::Renderer(Scene* pScene, int width, int height) {
	mScene = pScene;
	// Add a camera to the scene
	mCameraMat = new CameraMaterial(width, height);
	mCameraObject = new PlaneObject(mCameraMat);
	mScene->addObject(mCameraObject);
}

Renderer::~Renderer() {
	// TODO Remove our camera and delete it
}

void Renderer::doRenderPass(int photons) {

	// Fire a number of photons into the scene
	for(int i=0;i<photons;i++) {
		// pick a random light
		AbstractLight* light = mScene->getRandomLight();
		// Get a random ray
		Ray r = light->getRandomRayFromLight();
		// shoot
		RenderObject* obj = mScene->getClosestIntersection(&r);
	}
}

} /* namespace photonCPU */
