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
	float mColourWavelengthMin;
	float mColourWavelengthMax;
	photonCPU::NormalRandomGenerator* mRand;
	float randFloat();
public:
	float std;
	float ideal_range;
	ColourMaterial(float pColorWavelengthMin, float pColourWavelengthMax);
	virtual ~ColourMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength);
	#ifdef PHOTON_OPTIX
	    virtual optix::Material getOptiXMaterial(optix::Context context);
	#endif
};

}

#endif /* COLOURMATERIAL_H_ */
