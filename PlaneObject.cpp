/*
 * PlaneObject.cpp
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#include "PlaneObject.h"

namespace photonCPU {

PlaneObject::PlaneObject(AbstractMaterial* pMaterial) : RenderObject(pMaterial)
{
	position = new Vector3D();
	normal = new Vector3D(0, 0, 1);
	up = new Vector3D(0, 1, 0);
	right = new Vector3D(1, 0, 0);
	up->normaliseSelf();
	right->normaliseSelf();
}

PlaneObject::~PlaneObject() {
	delete position;
	delete normal;
}

void PlaneObject::setPosition(Vector3D* pos) {
	position->setTo(pos);
}
void PlaneObject::setPosition(float x, float y, float z) {
	position->setTo(x, y, z);
}
void PlaneObject::setNormal(Vector3D* n) {
	normal->setTo(n);
	normal->normaliseSelf();
}
void PlaneObject::setNormal(float x, float y, float z){
	normal->setTo(x, y, z);
	normal->normaliseSelf();
}

float PlaneObject::intersects(photonCPU::Ray* r) {
	float i = r->getDirection().dotProduct(normal);
	if(i==0) return 0;
	Vector3D adjusted_position = r->getPosition()-*position;
	//printf("[normal %f, %f, %f]", normal->x, normal->y, normal->z);
	return -adjusted_position.dotProduct(normal)/i;

}

/**
 * WARNING we allocate memory here
 *
 */
Vector3D PlaneObject::getIntersectionPoint(photonCPU::Ray* r) {
	float i = r->getDirection().dotProduct(normal);
	if(i!=0){
		Vector3D adjusted_position = r->getPosition()-*position;
		float t = -adjusted_position.dotProduct(normal)/i;
		return r->getPosition()+r->getDirection()*t;
	}
	// HURF DURF
	// Reutrn null? Somehow? I wish.
	return 0;
}

void PlaneObject::getTextureCordsAtPoint(photonCPU::Vector3D* point, float* u, float* v, float* w) {
	// Project onto our plane and take that, this should account for rounding errors
	// that result in points just off of our plane.
	Vector3D adjustedLoc = (*point-*position);
	float m = adjustedLoc.dotProduct(normal);
	Vector3D projection = (*point)-((*normal)*m);
	// right, convert this to local cordinates on the plane
	Vector3D diff = projection-(*position);
	*u = diff.dotProduct(up);
	*v = diff.dotProduct(right);
	*w = 0;
}

Ray* PlaneObject::transmitRay(Ray* r) {
	float u, v, w;
	Vector3D intersect = getIntersectionPoint(r);
	getTextureCordsAtPoint(&(intersect), &u, &v, &w);
	// Get our reflected ray
	Vector3D dir = r->getDirection();
	return mMaterial->transmitRay(&intersect, &dir, normal, u, v, w);
}

} /* namespace photonCPU */
