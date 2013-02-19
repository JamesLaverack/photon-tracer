/*
 * CameraMaterial.cpp
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#include "CameraMaterial.h"

namespace photonCPU {

CameraMaterial::CameraMaterial(int width, int height) {
	image = new float[width][height][3];
	imageWidth = width;
	imageHeight = height;
}

CameraMaterial::~CameraMaterial() {
	// TODO Auto-generated destructor stub
}

Ray CameraMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w) {
	// Are we in the renderable bit?


	return 0;
}

} /* namespace photonCPU */
