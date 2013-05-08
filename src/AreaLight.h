/*
 * AreaLight.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef AREALIGHT_H_
#define AREALIGHT_H_

#include <cstdlib>
#include "AbstractLight.h"

namespace photonCPU {

class AreaLight: public photonCPU::AbstractLight {
private:
	Vector3D* mPosition;
        Vector3D* mNormal;
        Vector3D* mUp;
        Vector3D* mRight;
	float mVariance;
        float mWidth, mHeight;
	float randFloat();
public:
	AreaLight(Vector3D* pPosition, Vector3D* pNormal, Vector3D* pUp, Vector3D* pRight, float pWidth, float pHeight, float pVaraince);
	virtual ~AreaLight();
	virtual Ray* getRandomRayFromLight();
	#ifdef PHOTON_OPTIX
		virtual optix::Program getOptiXLight(optix::Context context);
	#endif
};

} /* namespace photonCPU */
#endif /* AREALIGHT_H_ */
