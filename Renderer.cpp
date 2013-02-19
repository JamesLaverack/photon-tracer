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
	RenderObject* obj = 1; // assign to holder value
	// Fire a number of photons into the scene
	for(int i=0;i<photons;i++) {
		while(obj!=0) {
			// Pick a random light
			AbstractLight* light = mScene->getRandomLight();
			// Get a random ray
			Ray r = light->getRandomRayFromLight();
			// Shoot
			RenderObject* obj = mScene->getClosestIntersection(&r);
			// Did we hit anything?
			if (obj != 0) {
				// We hit some stuff. Decide what happens to our ray
				// get our intersection point
				Vector3D intersect = obj->getIntersectionPoint(&r);
				// get our texture coords
				int uvw[] = obj->getTextureCordsAtPoint(obj->getIntersectionPoint(&r));
				// Get our reflected ray
				Ray bounce = obj->getMaterial()->transmitRay(&intersect, &(r.getDirection()), normal, uvw[0], uvw[1], uvw[2]);
			}
		}
	}
}

/*
 * This method launches a random ray from a random light and finds whatever it hits.
 */
RenderObject* shootRayForObject() {

}

} /* namespace photonCPU */
