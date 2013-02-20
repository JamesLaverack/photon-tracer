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
#include "AbstractMaterial.h"

namespace photonCPU {

class CameraMaterial: public photonCPU::AbstractMaterial {
private:
	int imageWidth;
	int imageHeight;
	float* image;
	int vdiff;
	int udiff;
	int index(int x, int y, int z);
public:
	CameraMaterial(int width, int height);
	virtual ~CameraMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w);
	void toPPM();
};

} /* namespace photonCPU */
#endif /* CAMERAMATERIAL_H_ */
