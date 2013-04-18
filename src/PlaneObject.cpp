/*
 * PlaneObject.cpp
 *
 *  Created on: 6 Feb 2013
 *      Author: james
 */

#include "PlaneObject.h"

namespace photonCPU {

PlaneObject::PlaneObject(AbstractMaterial* pMaterial) : RenderObject(pMaterial)
{
	position = new Vector3D();
	normal = new Vector3D(0, 0, 1);
	up = new Vector3D(0, 1, 0);
	up->normaliseSelf();
	Vector3D tmp = normal->crossProduct(*up);
	tmp = tmp*-1;
	right = new Vector3D(0, 0, 0);
	right->setTo(&tmp);
	up->normaliseSelf();
	right->normaliseSelf();
	// This shit only matters for a camera, but whatever.
	focal_length = 26.0f;
	// Focal point is behind us
	Vector3D scaled_normal = ((*normal)*focal_length);
	focal_point = *position-scaled_normal;
	height = 100;
	width = 100;
}

PlaneObject::~PlaneObject() {
	delete position;
	delete normal;
	delete up;
	delete right;
}

void PlaneObject::setPosition(Vector3D* pos) {
	position->setTo(pos);
}
void PlaneObject::setPosition(float x, float y, float z) {
	position->setTo(x, y, z);
}
void PlaneObject::setNormal(Vector3D* n) {
	normal->setTo(n);
	normal->normaliseSelf();
}
void PlaneObject::setNormal(float x, float y, float z){
	normal->setTo(x, y, z);
	normal->normaliseSelf();
}

float PlaneObject::intersects(photonCPU::Ray* r) {
	float i = r->getDirection().dotProduct(normal);
	if(i==0) return 0;
	Vector3D adjusted_position = r->getPosition()-*position;
	//printf("[normal %f, %f, %f]", normal->x, normal->y, normal->z);
	float result =  -adjusted_position.dotProduct(normal)/i;
	return result;
}

/**
 * WARNING we allocate memory here
 *
 */
Vector3D PlaneObject::getIntersectionPoint(photonCPU::Ray* r) {
	float i = r->getDirection().dotProduct(normal);
	if(i!=0){
		Vector3D adjusted_position = r->getPosition()-*position;
		float t = -adjusted_position.dotProduct(normal)/i;
		return r->getPosition()+r->getDirection()*t;
	}
	// HURF DURF
	// Reutrn null? Somehow? I wish.
	return 0;
}

#ifdef PHOTON_OPTIX
optix::Geometry PlaneObject::getOptiXGeometry(optix::Context context) {
	optix::Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount( 1u );
	parallelogram->setBoundingBoxProgram( context->createProgramFromPTXFile("ptx/PlaneObject.ptx", "bounds" ) );
	parallelogram->setIntersectionProgram( context->createProgramFromPTXFile("ptx/PlaneObject.ptx", "intersect" ) );
	parallelogram["plane"]->setFloat(    normal->x,       normal->y,       normal->z, normal->dotProduct(position));
	// Move the anchor point
	Vector3D anchor = *position;
	anchor = anchor + (*up)*(-height/2);
	anchor = anchor + (*right)*(-width/2);
	parallelogram["anchor"]->setFloat( anchor.x,     anchor.y,     anchor.z );
	Vector3D v1 = *up*width;
	Vector3D v2 = *right*height;
	v1 = v1 * (1/(v1.dotProduct(&v1)));
	v2 = v2 * (1/(v2.dotProduct(&v2)));
	parallelogram["v1"]->setFloat(v1.x, v1.y, v1.z );
	parallelogram["v2"]->setFloat(v2.x, v2.y, v2.z );
	return parallelogram;
}
#endif

void PlaneObject::getTextureCordsAtPoint(photonCPU::Vector3D* point, float* u, float* v, float* w) {
	// Project onto our plane and take that, this should account for rounding errors
	// that result in points just off of our plane.
	Vector3D adjustedLoc = (*point)-(*position);
	float m = adjustedLoc.dotProduct(normal);
	Vector3D projection = (*point)-((*normal)*m);
	// right, convert this to local cordinates on the plane
	Vector3D diff = projection-(*position);
	*u = diff.dotProduct(right);
	*v = diff.dotProduct(up);
	*w = 0;
}

Ray* PlaneObject::transmitRay(Ray* r) {
	float u, v, w = 0;
	Vector3D intersect = getIntersectionPoint(r);
	getTextureCordsAtPoint(&(intersect), &u, &v, &w);
	// In bounds?
	if((std::abs(u)>(width/2)) | (std::abs(v)>(height/2))) {
		Ray* re = new Ray();
		Vector3D pos = r->getPosition();
		re->setPosition(&pos);
		Vector3D dir = r->getDirection();
		re->setDirection(&dir);
		re->wavelength = r->wavelength;
		return re;
	}

	// Get our reflected ray
	Vector3D dir = r->getDirection();
	/*
	printf("plane strike at (%f, %f, %f)\n", intersect.x, intersect.y, intersect.z );
	printf("   Ray had orgin (%f, %f, %f) and direction <%f, %f, %f>.\n",
			r->getPosition().x,
			r->getPosition().y,
			r->getPosition().z,
			r->getDirection().x,
			r->getDirection().y,
			r->getDirection().z
	);
	*/
	Vector3D psp_norm = intersect-focal_point;
	psp_norm.normaliseSelf();
	return mMaterial->transmitRay(&intersect, &dir, normal, &psp_norm, u, v, w, r->wavelength);
}

} /* namespace photonCPU */
