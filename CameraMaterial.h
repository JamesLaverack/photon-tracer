/*
 * CameraMaterial.h
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#ifndef CAMERAMATERIAL_H_
#define CAMERAMATERIAL_H_

#include "AbstractMaterial.h"

namespace photonCPU {

class CameraMaterial: public photonCPU::AbstractMaterial {
private:
	int imageWidth;
	int imageHeight;
	float* image[][][];
public:
	CameraMaterial(int width, int height);
	virtual ~CameraMaterial();
	Ray transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w);
};

} /* namespace photonCPU */
#endif /* CAMERAMATERIAL_H_ */
