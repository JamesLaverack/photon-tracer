/*
 * PointLight.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef POINTLIGHT_H_
#define POINTLIGHT_H_

#include <cstdlib>
#include "AbstractLight.h"

namespace photonCPU {

class PointLight: public photonCPU::AbstractLight {
private:
	Vector3D* mPosition;
	float randFloat();
public:
	PointLight(Vector3D* pPosition);
	PointLight(float px, float py, float pz);
	virtual ~PointLight();
	virtual Ray* getRandomRayFromLight();
	#ifdef PHOTON_OPTIX
		virtual optix::Program getOptiXLight(optix::Context context);
	#endif
};

} /* namespace photonCPU */
#endif /* POINTLIGHT_H_ */
