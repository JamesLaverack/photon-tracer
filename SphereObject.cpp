/*
 * SphereObject.cpp
 *
 *  Created on: Feb 21, 2013
 *      Author: James Laverack
 */

#include "SphereObject.h"

namespace photonCPU {

SphereObject::SphereObject(AbstractMaterial* pMaterial) : RenderObject(pMaterial) {
	position = new Vector3D();
	radius = 1;
}

SphereObject::~SphereObject() {
	delete position;
}

void SphereObject::setPosition(Vector3D* pos) {
	position->setTo(pos);
}
void SphereObject::setPosition(float x, float y, float z) {
	position->setTo(x, y, z);
}
void SphereObject::setRadius(float r) {
	radius = r;
}
float SphereObject::intersects(photonCPU::Ray* r) {
	Vector3D adjustedPos = (r->getPosition())-(*(position));
	float posDotDir = (r->getDirection()).dotProduct(&(adjustedPos));
	float posSquare = adjustedPos.dotProduct(&adjustedPos);

	float root = std::sqrt(posDotDir*posDotDir-posSquare+radius*radius);
	if(root < 0) {
		return 0;
	}

	float tmax = -posDotDir+root;
	float tmin = -posDotDir-root;

	if(tmax < tmin) {
		return tmax;
	} else {
		return tmin;
	}
}
Vector3D SphereObject::getIntersectionPoint(photonCPU::Ray* r) {
	return (r->getPosition())+(r->getDirection())*intersects(r);
}
void SphereObject::getTextureCordsAtPoint(photonCPU::Vector3D* point, float* u, float* v, float* w) {
	*u = 0;
	*v = 0;
	*w = 0;
}
Ray* SphereObject::transmitRay(Ray* r) {
	float u, v, w;
	Vector3D intersect = getIntersectionPoint(r);
	getTextureCordsAtPoint(&(intersect), &u, &v, &w);
	// Get our reflected ray
	Vector3D dir = r->getDirection();
	Vector3D normal = *(position)-intersect;
	return mMaterial->transmitRay(&intersect, &dir, &normal, u, v, w, r->wavelength);
}

}
