/*
 * TransparantMaterial.cpp
 *
 *  Created on: Mar 12, 2013
 *      Author: James Laverack
 */

#include "TransparantMaterial.h"

namespace photonCPU {

TransparantMaterial::TransparantMaterial() {
	// Glass
	index_of_refraction = 1.52f;
	radius = 10;
	debug_id = -1;
	lens_hack_radius = 10000;
	lens_hack_depth = 10000;

	b1 = 1.03961212f;
	b2 = 0.231792344f;
	b3 = 1.01046945f;
	c1 = 6.00069867e-3f;
	c2 = 2.00179144e-2f;
	c3 = 1.03560653e2f;
}

TransparantMaterial::~TransparantMaterial() {
	// TODO Auto-generated destructor stub
}
/*
Vector3D refract(Vector3D* angle, Vector3D* normal, Vector3D* axis, float index)div %f,  {
	float pi = 3.141;
	float angle_in = pi - std::acos(angle->dotProduct(normal));
	float div = std::sin(angle_in)/index;
	float angle_out = std::asin(div);
	float theta = angle_in - angle_out;
	//printf("in %f, out %f, theta %f, div %f, index %f\n",angle_in, angle_out, theta, div, index);
	Vector3D r;
	r = angle->rotate(axis, theta);
	return r;
}
*/

#ifdef PHOTON_OPTIX
optix::Material TransparantMaterial::getOptiXMaterial(optix::Context context) {
	optix::Program chp = context->createProgramFromPTXFile( "ptx/TransparentMaterial.ptx", "closest_hit" );
	optix::Material mat = context->createMaterial();
	mat["index_of_refraction"]->setFloat(index_of_refraction);
	mat["hack_lens_depth"]->setFloat(lens_hack_depth);
	mat["hack_lens_radius"]->setFloat(lens_hack_radius);
	mat["b_vals"]->setFloat(b1, b2, b3);
	mat["c_vals"]->setFloat(c1, c2, c3);
	mat["debug_id"]->setInt(debug_id);
	mat->setClosestHitProgram(0, chp);
	return mat;
}
#endif

Ray* TransparantMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	(void)perspective_normal;
	(void)u;
	(void)v;
	(void)w;

	// Some defintions
	const float pi = 3.141;

	bool report = false;
	if( (angle->x-0.584472) + (angle->y-0.009688) + (angle->z-0.811356) < 0.00001 ) {
		report = true;
	}
	report = false;

	// Calculate axis of rotation
	Vector3D axis = angle->crossProduct(*normal);
	axis.normaliseSelf();

	// Get our new direction vector
	Vector3D vec;

	// Angle between normal and the ray direction
	float angle_in = std::acos(angle->dotProduct(normal));

	// which side are we coming from
	float index = index_of_refraction;
	if(std::abs(hitLocation->z)>20) {
			index = 1;
	}
	if(std::sqrt(hitLocation->x*hitLocation->x + hitLocation->y*hitLocation->y) > radius) {
			index = 1;
	}

	if(angle_in < pi/2) {
		// We are coming from the material out
		index = 1/index;
	} else {
		// From the outside into the material
		angle_in = pi - angle_in;
	}
	
	float angle_out = std::asin(std::sin(angle_in)/index);
	float theta = -std::abs(angle_out - angle_in);
	//printf("in %f, out %f, theta %f, index %f\n",angle_in, angle_out, theta, index);
	vec = angle->rotate(&axis, theta);

	if(report) {
		printf("***\n");
		printf("axis ");
		axis.print();
		printf("angle in %f\n", angle_in);
		printf("angle out %f\n", angle_out);
		printf("theta %f\n", theta);
		printf("orginal direction ");
		angle->print();
		printf("current r %f\n", index);
		printf("bounce direction ");
		vec.print();
	}
	// Return new ray object
	Ray* r = new Ray();
	r->setPosition(hitLocation);
	r->wavelength = wavelength;
	r->setDirection(&vec);
	return r;
}


}
