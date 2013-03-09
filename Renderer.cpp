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
	// Remove the camera from the scene
	mScene->delObject(mCameraObject);
	// Delete the camera
	delete mCameraObject;
	delete mCameraMat;

}

void Renderer::doRenderPass(int photons) {
	Ray* r;
	const int maxBounce = 10;
	int bounceCount = 0;
	bool loop = true;
	printf("[Begin photon traciTheirng]\n");
	// Fire a number of photons into the scene
	for(int i=0;i<photons;i++) {

		//printf("<%d>\n", i);
		// Pick a random light
		AbstractLight* light = mScene->getRandomLight();
		// Get a random ray
		r = light->getRandomRayFromLight();
		int hundreth = (photons/100);
		if (photons%100>0) hundreth++;
		//printf("hundreth %d\n", hundreth);
		if((i % hundreth) == 0) {
			printf("    %d\n", (int) (100*(i/(float) photons)));
		}
		loop = true;
		bounceCount = 0;
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
			if ((obj != 0) && (bounceCount<maxBounce)) {
				bounceCount++;
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
				delete r;
		}
		}
		}
	mCameraMat->verify();
	//writeout
	mCameraMat->toPPM();
}

} /* namespace photonCPU */
