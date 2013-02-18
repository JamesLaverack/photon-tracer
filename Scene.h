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
#include "RenderObject.h"

namespace photonCPU {

class Scene {
private:
	std::vector<RenderObject*> mObjects;
public:
	Scene();
	virtual ~Scene();
	void addObject(RenderObject* obj);
	RenderObject* getClosestIntersection(Ray* r);
};

} /* namespace photonCPU */
#endif /* SCENE_H_ */
