/*
 * AbstractLight.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef ABSTRACTLIGHT_H_
#define ABSTRACTLIGHT_H_

#include "Ray.h"
#ifdef PHOTON_OPTIX
        #include <optixu/optixpp_namespace.h>
#endif

namespace photonCPU {

class AbstractLight {
public:
	AbstractLight();
	virtual ~AbstractLight();
	virtual photonCPU::Ray* getRandomRayFromLight() = 0;
	#ifdef PHOTON_OPTIX
		virtual optix::Program getOptiXLight(optix::Context context) = 0;
	#endif
};

} /* namespace photonCPU */
#endif /* ABSTRACTLIGHT_H_ */
