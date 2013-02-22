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
	float a = randFloat();
	float b   = randFloat();
	float phi = 2*3.141*a;
	float theta = std::acos(2*b-1);
	//printf("theta %f phi %f\n", theta, phi);
	float x, y, z;
	// Rotate<1, 0, 0> by theta around the y axis
	x = (float) std::sin(theta)*std::cos(phi);
	z = (float) std::sin(theta)*std::sin(phi);
	y = (float) std::cos(theta);

	//y = 0;
	//printf("<%f, %f, %f>\n", x, y, z);
	r->setDirection(x, y, z);
	r->wavelength = randFloat()*400+380;
	//r->getDirection().print();
	return r;
}



float PointLight::randFloat() {
	float f = (float)(rand()/((float)RAND_MAX));
	//printf("float %f\n", f);
	return f;
}

} /* namespace photonCPU */
