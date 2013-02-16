/*
 * Vector3D.cpp
 *
 *  Created on: 11 Feb 2013
 *      Author: james
 */

#include "Vector3D.h"

namespace photonCPU {

Vector3D::Vector3D() {
	// init values
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
}

Vector3D::~Vector3D() {
	// TODO Auto-generated destructor stub
}

float Vector3D::dotProduct(Vector3D b) {
	return x*b.x+y*b.y+z*b.z;
}

/**
 * Tells the vector to normalise itself. THIS IS DONE IN PLACE.
 */
void Vector3D::normaliseSelf() {
	// Sum
	float sum = x+y+z;
	if(sum!=0){
		x = x/sum;
		y = y/sum;
		z = z/sum;
	}
}

} /* namespace photonCPU */

