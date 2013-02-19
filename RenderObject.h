/*
 * RenderObject.h
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#ifndef RENDEROBJECT_H_
#define RENDEROBJECT_H_

#include "Ray.h"
#include "AbstractMaterial.h"

namespace photonCPU {

class RenderObject {
protected:
	AbstractMaterial* mMaterial;
public:
	RenderObject(AbstractMaterial* pMaterial);
	virtual ~RenderObject();
	AbstractMaterial* getMaterial();
	virtual float intersects(photonCPU::Ray* r) = 0;
	virtual Vector3D getIntersectionPoint(photonCPU::Ray* r) = 0;
	virtual int* getTextureCordsAtPoint(photonCPU::Vector3D* point) = 0;
};

} /* namespace photonCPU */
#endif /* RENDEROBJECT_H_ */
