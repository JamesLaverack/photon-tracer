/*
 * Scene.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include <limits>
#include <cstdlib>
#include <algorithm>
#include "RenderObject.h"
#include "AbstractLight.h"

namespace photonCPU {

class Scene {
public:
	std::vector<RenderObject*> mObjects;
	std::vector<AbstractLight*> mLights;
	Scene();
	virtual ~Scene();
	void addObject(RenderObject* obj);
	void delObject(RenderObject* obj);
	void addLight(AbstractLight* light);
	void delLight(AbstractLight* light);
	RenderObject* getClosestIntersection(Ray* r);
	AbstractLight* getRandomLight();
};

} /* namespace photonCPU */
#endif /* SCENE_H_ */
