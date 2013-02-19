/*
 * PerfectMirrorMaterial.h
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#ifndef PERFECTMIRRORMATERIAL_H_
#define PERFECTMIRRORMATERIAL_H_

#include "AbstractMaterial.h"

namespace photonCPU {

class PerfectMirrorMaterial: public photonCPU::AbstractMaterial {
public:
	PerfectMirrorMaterial();
	virtual ~PerfectMirrorMaterial();
	Ray transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w);
};

} /* namespace photonCPU */
#endif /* PERFECTMIRRORMATERIAL_H_ */
