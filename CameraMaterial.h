/*
 * CameraMaterial.h
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#ifndef CAMERAMATERIAL_H_
#define CAMERAMATERIAL_H_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "AbstractMaterial.h"
#include "WavelengthToRGB.h"

namespace photonCPU {

class CameraMaterial: public photonCPU::AbstractMaterial {
private:
	int imageWidth;
	int imageHeight;
	float actualWidth;
	float actualHeight;
	float *imageR;
	float *imageG;
	float *imageB;
	int vdiff;
	int udiff;
	bool derp;
	int index(int x, int y);
	photonCPU::WavelengthToRGB *converter;
	void initImage();
public:
	CameraMaterial(int width, int height);
	virtual ~CameraMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w, float wavelength);
	void toPPM();
	void verify();
};

} /* namespace photonCPU */
#endif /* CAMERAMATERIAL_H_ */
