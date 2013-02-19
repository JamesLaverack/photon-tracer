/*
 * Ray.h
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#include "Vector3D.h"

#ifndef RAY_H_
#define RAY_H_

namespace photonCPU {

class Ray {
private:
	Vector3D* position;
	Vector3D* direction;
public:
	Ray();
	virtual ~Ray();
	void setPosition(Vector3D* pos);
	void setPosition(float x, float y, float z);
	Vector3D getPosition();
	void setDirection(Vector3D* normal);
	void setDirection(float x, float y, float z);
	Vector3D getDirection();
};

} /* namespace photonCPU */
#endif /* RAY_H_ */
