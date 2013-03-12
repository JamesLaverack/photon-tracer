/*
 * RadiusMaskMaterial.h
 *
 *  Created on: Mar 12, 2013
 *      Author: James Laverack
 */

#ifndef RADIUSMASKMATERIAL_H_
#define RADIUSMASKMATERIAL_H_

#include <cmath>
#include "AbstractMaterial.h"

namespace photonCPU {

class RadiusMaskMaterial: public photonCPU::AbstractMaterial {
public:
	float radius;
	RadiusMaskMaterial();
	virtual ~RadiusMaskMaterial();
	virtual Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength);
};

}

#endif /* RADIUSMASKMATERIAL_H_ */
