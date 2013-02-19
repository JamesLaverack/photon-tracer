/*
 * PointLight.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "PointLight.h"

namespace photonCPU {

PointLight::PointLight(Vector3D* pPosition) {
	mPosition = new Vector3D(pPosition);
}

PointLight::PointLight(float px, float py, float pz) {
	mPosition = new Vector3D(px, py, pz);
}

PointLight::~PointLight() {
	delete mPosition;
}

Ray PointLight::getRandomRayFromLight() {
	// Point light, so always emmit from 1 place
	Ray r = new Ray();
	r.setPosition(mPosition);
	// Randomise direction
	r.setDirection(randFloat(), randFloat(), randFloat());
}

float PointLight::randFloat() {
	return (float)rand()/(float)RAND_MAX;
}

} /* namespace photonCPU */
