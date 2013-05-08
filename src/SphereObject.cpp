/*
 * SphereObject.cpp
 *
 *  Created on: Feb 21, 2013
 *      Author: James Laverack
 */

#include "SphereObject.h"

namespace photonCPU {

SphereObject::SphereObject(AbstractMaterial* pMaterial) : RenderObject(pMaterial) {
	position = new Vector3D();
	radius = 10;
}

SphereObject::~SphereObject() {
	delete position;
}

void SphereObject::setPosition(Vector3D* pos) {
	position->setTo(pos);
}
void SphereObject::setPosition(float x, float y, float z) {
	position->setTo(x, y, z);
}
void SphereObject::setRadius(float r) {
	radius = r;
}
float SphereObject::intersects(photonCPU::Ray* r) {
	Vector3D adjustedPos = (r->getPosition())-(*(position));
	float posDotDir = (r->getDirection()).dotProduct(&(adjustedPos));
	float posSquare = adjustedPos.dotProduct(&adjustedPos);

	float innerRoot = posDotDir*posDotDir-posSquare+radius*radius;
	if(innerRoot <= 0) {
		return 0;
	}

	float root = std::sqrt(innerRoot);

	float tmax = -posDotDir+root;
	float tmin = -posDotDir-root;

	if(tmax < tmin) {
		return tmax;
	} else {
		return tmin;
	}
}
Vector3D SphereObject::getIntersectionPoint(photonCPU::Ray* r) {
	return (r->getPosition())+(r->getDirection())*intersects(r);
}
void SphereObject::getTextureCordsAtPoint(photonCPU::Vector3D* point, float* u, float* v, float* w) {
	*u = 0;
	*v = 0;
	*w = 0;
}
#ifdef PHOTON_OPTIX
optix::Geometry SphereObject::getOptiXGeometry(optix::Context context) {
	optix::Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount( 1u );
	sphere->setBoundingBoxProgram( context->createProgramFromPTXFile("ptx/SphereObject.ptx", "bounds" ) );
	sphere->setIntersectionProgram( context->createProgramFromPTXFile("ptx/SphereObject.ptx", "robust_intersect" ) );
	sphere["sphere"]->setFloat( position->x, position->y, position->z, radius );
	return sphere;
}
#endif
Ray* SphereObject::transmitRay(Ray* r) {
	float u, v, w;
	Vector3D intersect = getIntersectionPoint(r);
	getTextureCordsAtPoint(&(intersect), &u, &v, &w);
	// Get our reflected ray
	Vector3D dir = r->getDirection();
	Vector3D normal = *(position)-intersect;
	normal.normaliseSelf();
	/*
	printf("sphere strike at (%f, %f, %f)\n", intersect.x, intersect.y, intersect.z );
	printf("   Ray had orgin (%f, %f, %f) and direction <%f, %f, %f>.\n",
			r->getPosition().x,
			r->getPosition().y,
			r->getPosition().z,
			r->getDirection().x,
			r->getDirection().y,
			r->getDirection().z
	);
	*/
	return mMaterial->transmitRay(&intersect, &dir, &normal, &normal, u, v, w, r->wavelength);
}

}
