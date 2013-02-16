/*
 * RenderObject.h
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#ifndef RENDEROBJECT_H_
#define RENDEROBJECT_H_

#include "Ray.h"

namespace photonCPU {

class RenderObject {
public:
	RenderObject();
	virtual ~RenderObject();
	virtual bool intersects(photonCPU::Ray r) = 0;
};

} /* namespace photonCPU */
#endif /* RENDEROBJECT_H_ */
