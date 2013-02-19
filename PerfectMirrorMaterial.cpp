/*
 * PerfectMirrorMaterial.cpp
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#include "PerfectMirrorMaterial.h"

namespace photonCPU {

PerfectMirrorMaterial::PerfectMirrorMaterial() {
	// TODO Auto-generated constructor stub

}

PerfectMirrorMaterial::~PerfectMirrorMaterial() {
	// TODO Auto-generated destructor stub
}

Ray PerfectMirrorMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w) {
	Ray r;
	r.setPosition(hitLocation);
	// Do perfect reflection about the normal.
	Vector3D i = (-1.0f)*angle;
	float m = 2*i.dotProduct(normal);
	Vector3D reflection = (m*normal)-i;
	r.setDirection(&reflection);
	return r;
}

} /* namespace photonCPU */
