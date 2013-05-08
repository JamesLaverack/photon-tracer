/*
 * PerfectMirrorMaterial.h
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#ifndef PERFECTMIRRORMATERIAL_H_
#define PERFECTMIRRORMATERIAL_H_

#include "AbstractMaterial.h"
#include <cstdlib>

namespace photonCPU {

class PerfectMirrorMaterial: public photonCPU::AbstractMaterial {
public:
	PerfectMirrorMaterial();
	virtual ~PerfectMirrorMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength);
	#ifdef PHOTON_OPTIX
	    virtual optix::Material getOptiXMaterial(optix::Context context);
	#endif
};

} /* namespace photonCPU */
#endif /* PERFECTMIRRORMATERIAL_H_ */
