/*
 * PerfectMattMaterial.h
 *
 *  Created on: Feb 21, 2013
 *      Author: James Laverack
 */

#ifndef PERFECTMATTMATERIAL_H_
#define PERFECTMATTMATERIAL_H_

#include "AbstractMaterial.h"

namespace photonCPU {

class PerfectMattMaterial: public photonCPU::AbstractMaterial {
public:
	PerfectMattMaterial();
	virtual ~PerfectMattMaterial();
	virtual Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w);
};

} /* namespace photonCPU */

#endif /* PERFECTMATTMATERIAL_H_ */
