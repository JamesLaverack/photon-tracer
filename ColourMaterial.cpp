/*
 * ColourMaterial.cpp
 *
 *  Created on: Mar 2, 2013
 *      Author: James Laverack
 */

#include "ColourMaterial.h"

namespace photonCPU {

ColourMaterial::ColourMaterial(float pColourWavelengthMin, float pColourWavelengthMax) {
	mColourWavelengthMax = pColourWavelengthMax;
	mColourWavelengthMin = pColourWavelengthMin;
	mRand = new photonCPU::NormalRandomGenerator();
}

ColourMaterial::~ColourMaterial() {
	delete mRand;
}

Ray* ColourMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	// We don't care where we hit this material in terms of textures
	(void)u;
	(void)v;
	(void)w;
	(void)perspective_normal;

	// Do we absorb this?
	if(wavelength>mColourWavelengthMax) return 0;
	if(wavelength<mColourWavelengthMin) return 0;

	float std = 2;
	// Create new ray
	Ray* r = new Ray();
	r->setPosition(hitLocation);
	// Do perfect reflection about the normal.
	//// Vector3D i = (*angle)*(-1.0f);
	//// float m = i.dotProduct(normal)*2;
	//// Vector3D reflection = ((*normal)*m)-i;
	//Vector3D adjusted_angle = (*angle)*-1;
	float reflect_angle = std::acos(angle->dotProduct(normal));
	reflect_angle = reflect_angle - 3.141/2; // DEBUG CODE ONLY
	//printf("Reflection angle %f\n", reflect_angle);
	//reflect_angle = -reflect_angle;
	// project our incident ray onto the plane defined by
	// < hitLocation, normal > and make sure it's a unit vector, this becomes u.
	//Vector3D adjustedLoc = (*angle)-(*hitLocation);
	float m = angle->dotProduct(normal);
	Vector3D u_vec = (*angle)-((*normal)*m);
	//u_vec.setTo(0, angle->y, angle->z); // DEBUG CODE ONLY
	u_vec.normaliseSelf();
	// Calculate v from the cross product of u and normal
	Vector3D v_vec = u_vec.crossProduct(*normal);
	//printf("v (pre normalize)\n");
	//v_vec.print();
	v_vec.normaliseSelf();
	//printf("u\n");

	// get theta, which is the angle between our bounce and the normal in the u direction.
	// Also get phi, the angle between our bounce and the normal in the v direction.
	float theta = mRand->getRandom(reflect_angle, std);
	float phi   = mRand->getRandom(0            , std);
	//printf("Refection angle %f, theta %f, phi %f.\n", reflect_angle, theta, phi);
	// Construct our bounce vector, this is our actual reflection.
	Vector3D bounce;
	bounce.setTo(normal);
	// rotate by phi around the u vector.
	//printf("start <%f, %f, %f>\n", bounce.x, bounce.y, bounce.z);
	bounce = bounce.rotate(&u_vec, phi);
	// rotate by theta around the v vector.
	// (This order is correct, just think about it.)
	bounce = bounce.rotate(&v_vec, theta);
	bounce.normaliseSelf();
	//printf("end   <%f, %f, %f>\n	", bounce.x, bounce.y, bounce.z);
	float true_angle = std::acos(bounce.dotProduct(normal));

	/*
	printf("#\n");
	printf("reflection angle %f true angle %f\n", reflect_angle, true_angle);
	angle->print();
	u_vec.print();
	v_vec.print();
	bounce.print();
	*/

	r->setDirection(&bounce);
	r->wavelength = wavelength;
	return r;
}

}