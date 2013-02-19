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
	delete mMaterial;
}

} /* namespace photonCPU */
