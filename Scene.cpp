/*
 * Scene.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "Scene.h"

namespace photonCPU {

Scene::Scene() {
	// TODO Auto-generated constructor stub

}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

void Scene::addObject(RenderObject* obj) {
	mObjects.push_back(obj);
}

RenderObject* Scene::getClosestIntersection(Ray* r) {
	// Record smallest t
	float min_t = std::numeric_limits<float>::infinity();
	RenderObject* closest_object;
	// iterate
	for(std::vector<RenderObject*>::size_type i = 0; i != mObjects.size(); i++) {
		float t = ((RenderObject*) mObjects[i])->intersects(r);
		if ((t<min_t)&&(t>0)) {
			min_t = t;
			closest_object = mObjects[i];
		}
	}
	return closest_object;
}

} /* namespace photonCPU */
