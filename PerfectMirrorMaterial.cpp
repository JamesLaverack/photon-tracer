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

float randFloat() {
	float f = (float)(rand()/((float)RAND_MAX));
	//printf("float %f\n", f);
	return f;
}

Ray* PerfectMirrorMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w) {
	if(randFloat()<0.1f) {
		return 0;
	}
	Ray* r = new Ray();
	r->setPosition(hitLocation);
	// Do perfect reflection about the normal.
	Vector3D i = (*angle)*(-1.0f);
	float m = i.dotProduct(normal)*2;
	Vector3D reflection = ((*normal)*m)-i;
	r->setDirection(&reflection);
	return r;
}

} /* namespace photonCPU */
