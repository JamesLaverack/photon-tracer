/*
 * ColourMaterial.h
 *
 *  Created on: Mar 2, 2013
 *      Author: James Laverack
 */

#ifndef COLOURMATERIAL_H_
#define COLOURMATERIAL_H_

#include "AbstractMaterial.h"
#include "NormalRandomGenerator.h"

namespace photonCPU {

class ColourMaterial: public photonCPU::AbstractMaterial {
private:
	float mColourWavelength;
	photonCPU::NormalRandomGenerator* mRand;
public:
	ColourMaterial(float pColorWavelength);
	virtual ~ColourMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w, float wavelength);
};

}

#endif /* COLOURMATERIAL_H_ */
