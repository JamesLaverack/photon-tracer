/*
 * Ray.cpp
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#include "Ray.h"

namespace photonCPU {


Ray::Ray() {
	position = new Vector3D();
	direction = new Vector3D();
}

Ray::~Ray() {
	delete position;
	delete direction;
}

} /* namespace photonCPU */
