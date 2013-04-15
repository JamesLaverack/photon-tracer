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

#ifdef PHOTON_OPTIX
optix::Material RadiusMaskMaterial::getOptiXMaterial(optix::Context context) {
	optix::Program chp = context->createProgramFromPTXFile( "ptx/RadiusMaskMaterial.ptx", "closest_hit" );
	optix::Material mat = context->createMaterial();
	mat->setClosestHitProgram(0, chp);
	return mat;
}
#endif

Ray* RadiusMaskMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	(void)normal;
	(void)perspective_normal;
	(void)w;
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
