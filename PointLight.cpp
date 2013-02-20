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

Ray* PointLight::getRandomRayFromLight() {
	// Point light, so always emmit from 1 place
	Ray* r = new Ray();
	r->setPosition(mPosition);
	//printf("HURF\n");
	//r->getPosition().print();
	//r->getPosition().print();
	//printf("DURF\n");
	// Randomise direction
	r->setDirection(randFloat(), randFloat(), randFloat());
	return r;
}

float PointLight::randFloat() {
	return (float)(rand()-(RAND_MAX/2))/(float)(RAND_MAX/2);
}

} /* namespace photonCPU */
