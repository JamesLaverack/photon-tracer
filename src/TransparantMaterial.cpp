/*
 * TransparantMaterial.cpp
 *
 *  Created on: Mar 12, 2013
 *      Author: James Laverack
 */

#include "TransparantMaterial.h"

namespace photonCPU {

TransparantMaterial::TransparantMaterial() {
	// Glass
	index_of_refraction = 1.52f;
	radius = 10;
}

TransparantMaterial::~TransparantMaterial() {
	// TODO Auto-generated destructor stub
}
/*
Vector3D refract(Vector3D* angle, Vector3D* normal, Vector3D* axis, float index)div %f,  {
	float pi = 3.141;
	float angle_in = pi - std::acos(angle->dotProduct(normal));
	float div = std::sin(angle_in)/index;
	float angle_out = std::asin(div);
	float theta = angle_in - angle_out;
	//printf("in %f, out %f, theta %f, div %f, index %f\n",angle_in, angle_out, theta, div, index);
	Vector3D r;
	r = angle->rotate(axis, theta);
	return r;
}
*/
Ray* TransparantMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	(void)perspective_normal;
	(void)u;
	(void)v;
	(void)w;

	// Some defintions
	const float pi = 3.141;


	// Calculate axis of rotation
	Vector3D axis = angle->crossProduct(*normal);
	axis.normaliseSelf();

	// Get our new direction vector
	Vector3D vec;

	// Angle between normal and the ray direction
	float angle_in = std::acos(angle->dotProduct(normal));

	// which side are we coming from
	float index = index_of_refraction;
	if(std::abs(hitLocation->z)>20) {
			index = 1;
	}
	if((hitLocation->x)*(hitLocation->x) + (hitLocation->y)*(hitLocation->y) > (40*40)) {
              index = 1;
        }

	/*if(std::sqrt(hitLocation->x*hitLocation->x + hitLocation->y*hitLocation->y) > radius) {
			return 0;
	}*/
	if(angle_in < pi/2) {
		// We are coming from the material out
		index = 1/index;
	} else {
		// From the outside into the material
		angle_in = pi - angle_in;
	}

	float angle_out = std::asin(std::sin(angle_in)/index);
	float theta = angle_in - angle_out;
	//printf("in %f, out %f, theta %f, index %f\n",angle_in, angle_out, theta, index);
	vec = angle->rotate(&axis, -theta);

	// Return new ray object
	Ray* r = new Ray();
	r->setPosition(hitLocation);
	r->wavelength = wavelength;
	r->setDirection(&vec);
	return r;
}


}
