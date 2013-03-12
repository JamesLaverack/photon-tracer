/*
 * RadiusMaskMaterial.cpp
 *
 *  Created on: Mar 12, 2013
 *      Author: James Laverack
 */

#include "RadiusMaskMaterial.h"

namespace photonCPU {

RadiusMaskMaterial::RadiusMaskMaterial() {
	// TODO Auto-generated constructor stub

}

RadiusMaskMaterial::~RadiusMaskMaterial() {
	// TODO Auto-generated destructor stub
}

Ray* RadiusMaskMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {

	if(std::sqrt(u*u + v*v) > radius) {
		return 0;
	}
	Ray* r = new Ray();
	r->setPosition(hitLocation);
	r->wavelength = wavelength;
	r->setDirection(angle);
	return r;
}

}
