/*
 * AreaLight.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "AreaLight.h"

namespace photonCPU {

AreaLight::AreaLight(Vector3D* pPosition, Vector3D* pNormal, Vector3D* pUp, Vector3D* pRight, float pWidth, float pHeight) {
	mPosition = new Vector3D(pPosition);
        mNormal   = new Vector3D(pNormal);
        mUp       = new Vector3D(pUp);
        mRight    = new Vector3D(pRight);
        mWidth    = pWidth;
        mHeight   = pHeight;
	pPosition->print();
	printf("Created area light with position ");
	mPosition->print();
}

AreaLight::~AreaLight() {
	delete mPosition;
        delete mNormal;
        delete mUp;
        delete mRight;
}

Ray* AreaLight::getRandomRayFromLight() {
	// Point light, so always emmit from 1 place
	const float pi = 3.141;

        Vector3D position;
        position.setTo(mPosition);
        position += (*mRight)*(randFloat()*mWidth);
        position += (*mUp)*(randFloat()*mHeight);

        Vector3D bounce;
        float phi = randFloat()*pi - pi/2;
        float theta = randFloat()*pi - pi/2;
        bounce.setTo(mNormal);
        bounce = bounce.rotate(mUp, phi);
        bounce = bounce.rotate(mRight, theta);
        bounce.normaliseSelf();

        Ray* r = new Ray();
        r->setPosition(&position);
	r->setDirection(&bounce);
	r->wavelength = randFloat()*400+380;

	return r;
}

#ifdef PHOTON_OPTIX
optix::Program AreaLight::getOptiXLight(optix::Context context) {
	optix::Program prog = context->createProgramFromPTXFile( "ptx/AreaLight.ptx", "light" );
	prog["location"]->setFloat( mPosition->x, mPosition->y, mPosition->z);
        prog["normal"]->setFloat( mNormal->x, mNormal->y, mNormal->z);
        prog["up"]->setFloat( mUp->x, mUp->y, mUp->z);
        prog["right"]->setFloat( mRight->x, mRight->y, mRight->z);
        prog["width"]->setFloat(mWidth);
        prog["height"]->setFloat(mHeight);
	return prog;
}
#endif

float AreaLight::randFloat() {
	float f = (float)(rand()/((float)RAND_MAX));
	return f;
}

} /* namespace photonCPU */
