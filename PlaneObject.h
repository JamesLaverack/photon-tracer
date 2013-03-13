/*
 * PlaneObject.h
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#ifndef PLANEOBJECT_H_
#define PLANEOBJECT_H_

#include "RenderObject.h"
#include "Vector3D.h"

namespace photonCPU {

class PlaneObject: public photonCPU::RenderObject {
private:
	Vector3D* position;
	Vector3D* normal;
	Vector3D* up;
	Vector3D* right;
	float focal_length;
	Vector3D focal_point;
	int height;
	int width;
public:
	PlaneObject(AbstractMaterial* pMaterial);
	virtual ~PlaneObject();
	void setPosition(Vector3D* pos);
	void setPosition(float x, float y, float z);
	void setNormal(Vector3D* normal);
	void setNormal(float x, float y, float z);
	virtual float intersects(photonCPU::Ray* r);
	virtual Vector3D getIntersectionPoint(photonCPU::Ray* r);
	virtual void getTextureCordsAtPoint(photonCPU::Vector3D* point, float* u, float* v, float* w);
	virtual Ray* transmitRay(Ray* r);
};

} /* namespace photonCPU */
#endif /* PLANEOBJECT_H_ */
