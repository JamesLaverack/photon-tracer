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

void Scene::delObject(RenderObject* obj) {
	mObjects.erase(std::remove(vec.begin(), vec.end(), obj), vec.end());
}

void Scene::addLight(AbstractLight* light) {
	mLights.push_back(light);
}

void Scene::delLight(AbstractLight* light) {
	mLights.erase(std::remove(vec.begin(), vec.end(), light), vec.end());
}

RenderObject* Scene::getClosestIntersection(Ray* r) {
	// Record smallest t
	float min_t = std::numeric_limits<float>::infinity();
	RenderObject* closest_object = 0;
	// iterate
	for(std::vector<RenderObject*>::size_type i = 0; i != mObjects.size(); i++) {
		float t = ((RenderObject*) mObjects[i])->intersects(r);
		//printf("(t %f)", t);
		if ((t<min_t)&&(t>0.01)) {
			min_t = t;
			closest_object = mObjects[i];
		}
	}
	return closest_object;
}

AbstractLight* Scene::getRandomLight() {
	return mLights[rand() % mLights.size()];
}


} /* namespace photonCPU */
