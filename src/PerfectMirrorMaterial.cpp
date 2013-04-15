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

#ifdef PHOTON_OPTIX
optix::Material PerfectMirrorMaterial::getOptiXMaterial(optix::Context context) {
		optix::Program chp = context->createProgramFromPTXFile( "ptx/PerfectMirrorMaterial.ptx", "closest_hit" );
	optix::Material mat = context->createMaterial();
	mat->setClosestHitProgram(0, chp);
	return mat;
}
#endif

Ray* PerfectMirrorMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	// We don't care where we hit this material in terms of textures
	(void)u;
	(void)v;
	(void)w;
	(void)perspective_normal;

	// Create new ray
	Ray* r = new Ray();
	r->setPosition(hitLocation);
	// Do perfect reflection about the normal.
	Vector3D i = (*angle)*(-1.0f);
	float m = i.dotProduct(normal)*2;
	Vector3D reflection = ((*normal)*m)-i;
	r->setDirection(&reflection);
	r->wavelength = wavelength;
	return r;
}

} /* namespace photonCPU */
