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
#include "Image.h"

namespace photonCPU {

class CameraMaterial: public photonCPU::AbstractMaterial {
private:
	int imageWidth;
	int imageHeight;
	float actualWidth;
	float actualHeight;
	int vdiff;
	int udiff;
	bool derp;
	int fileid;
	photonCPU::WavelengthToRGB *converter;
	photonCPU::Image *img;
	float focalLength;
	float apatureSize;
	Vector3D* focalPoint;
public:
	CameraMaterial(int width, int height);
	virtual ~CameraMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength);
	Image* getImage();
};

} /* namespace photonCPU */
#endif /* CAMERAMATERIAL_H_ */
