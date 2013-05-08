/*
 * TransparantMaterial.h
 *
 *  Created on: Mar 12, 2013
 *      Author: James Laverack
 */

#ifndef TRANSPARANTMATERIAL_H_
#define TRANSPARANTMATERIAL_H_

#include "AbstractMaterial.h"

namespace photonCPU {

class TransparantMaterial: public photonCPU::AbstractMaterial {
private:
	float index_of_refraction;
public:
	float lens_hack_depth;
	float lens_hack_radius;
	float radius;
	int debug_id;
	TransparantMaterial();
	virtual ~TransparantMaterial();
	Ray* transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength);
	#ifdef PHOTON_OPTIX
	    virtual optix::Material getOptiXMaterial(optix::Context context);
	#endif
};

}

#endif /* TRANSPARANTMATERIAL_H_ */
