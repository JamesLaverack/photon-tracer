/*
 * AbstractMaterial.h
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#ifndef ABSTRACTMATERIAL_H_
#define ABSTRACTMATERIAL_H_

#include "Ray.h"
#ifdef PHOTON_OPTIX
	#include <optixu/optixpp_namespace.h>
#endif

namespace photonCPU {

class AbstractMaterial {
public:
	AbstractMaterial();
	virtual ~AbstractMaterial();
	virtual Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) = 0;
	#ifdef PHOTON_OPTIX
	    virtual optix::Material getOptiXMaterial(optix::Context context) = 0;
	#endif
};

} /* namespace photonCPU */
#endif /* ABSTRACTMATERIAL_H_ */
