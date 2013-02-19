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
	direction = new Vector3D(0, 0, 1);
}

Ray::~Ray() {
	delete position;
	delete direction;
}

void Ray::setPosition(Vector3D* pos) {
	position->setTo(pos);
}
void Ray::setPosition(float x, float y, float z) {
	position->setTo(x, y, z);
}
Vector3D Ray::getPosition() {
	Vector3D p = position;
	return p;
}
void Ray::setDirection(Vector3D* n) {
	direction->setTo(n);
	direction->normaliseSelf();
}
void Ray::setDirection(float x, float y, float z){
	direction->setTo(x, y, z);
	direction->normaliseSelf();
}
Vector3D Ray::getDirection() {
	Vector3D d = direction;
	return d;
}

} /* namespace photonCPU */
