/*
 * RenderObject.cpp
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#include "RenderObject.h"

namespace photonCPU {

RenderObject::RenderObject(AbstractMaterial* pMaterial) {
	mMaterial = pMaterial;
}

RenderObject::~RenderObject() {
}

AbstractMaterial* RenderObject::getMaterial() {
	return mMaterial;
}

#ifdef PHOTON_OPTIX
optix::Material RenderObject::getOptiXMaterial(optix::Context context) {
	return mMaterial->getOptiXMaterial(context);
}
#endif

} /* namespace photonCPU */
