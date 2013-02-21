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
	Ray* r;
	bool loop = true;
	// Fire a number of photons into the scene
	for(int i=0;i<photons;i++) {
		//printf("<%d>\n", i);
		// Pick a random light
		AbstractLight* light = mScene->getRandomLight();
		// Get a random ray
		r = light->getRandomRayFromLight();
		loop = true;
		while(loop) {
			loop = false;
			// debug
			//printf("    dir: ");
			//r->getDirection().print();
			//printf("    pos: ");
			//r->getPosition().print();
			// Shoot
			RenderObject* obj = mScene->getClosestIntersection(r);
			// Did we hit anything?
			if (obj != 0) {
				Ray* newr = obj->transmitRay(r);
				delete r;
				if(newr!=0) {
					//printf(" rly");
					r = newr;
					loop = true;
				}
				//printf(" STRIKE\n");
			}else{
				//printf(" MISS\n");
			}
		}
	}
	Ray* butts = new Ray();
	butts->setDirection(0, 0, -1);
	butts->setPosition(-10, 40, 10);
	mCameraObject->transmitRay(butts);
	delete butts;
	//writeout
	mCameraMat->toPPM();
}

} /* namespace photonCPU */
