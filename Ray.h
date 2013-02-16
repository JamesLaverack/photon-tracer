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

public:
    Vector3D* position;
    Vector3D* direction;
	Ray();
	virtual ~Ray();
};

} /* namespace photonCPU */
#endif /* RAY_H_ */
