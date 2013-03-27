/*
 * SphereObject.h
 *
 *  Created on: Feb 21, 2013
 *      Author: James Laverack
 */

#ifndef SPHEREOBJECT_H_
#define SPHEREOBJECT_H_

#include "RenderObject.h"
#include "AbstractMaterial.h"

namespace photonCPU {

class SphereObject: public photonCPU::RenderObject {
private:
	Vector3D* position;

public:
	float radius;
	SphereObject(AbstractMaterial* pMaterial);
	virtual ~SphereObject();
	void setPosition(Vector3D* pos);
	void setPosition(float x, float y, float z);
	void setRadius(float r);
	virtual float intersects(photonCPU::Ray* r);
	virtual Vector3D getIntersectionPoint(photonCPU::Ray* r);
	virtual void getTextureCordsAtPoint(photonCPU::Vector3D* point, float* u, float* v, float* w);
	virtual Ray* transmitRay(Ray* r);
};

}

#endif /* SPHEREOBJECT_H_ */
