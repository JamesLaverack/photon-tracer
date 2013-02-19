/*
 * PlaneObject.cpp
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#include "PlaneObject.h"

namespace photonCPU {

PlaneObject::PlaneObject(AbstractMaterial* pMaterial)
: RenderObject(pMaterial)
{
	position = new Vector3D();
	normal = new Vector3D(0, 1, 0);
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
	return r->getDirection().dotProduct(normal);
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
}

} /* namespace photonCPU */
