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
public:
	PlaneObject();
	virtual ~PlaneObject();
	void setPosition(Vector3D* pos);
	void setPosition(float x, float y, float z);
	void setNormal(Vector3D* normal);
	void setNormal(float x, float y, float z);
	virtual float intersects(photonCPU::Ray* r);
	virtual Vector3D getIntersectionPoint(photonCPU::Ray* r);
};

} /* namespace photonCPU */
#endif /* PLANEOBJECT_H_ */
