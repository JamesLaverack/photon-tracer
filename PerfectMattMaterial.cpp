/*
 * PerfectMattMaterial.cpp
 *
 *  Created on: Feb 21, 2013
 *      Author: James Laverack
 */

#include "PerfectMattMaterial.h"

namespace photonCPU {

PerfectMattMaterial::PerfectMattMaterial() {
	// TODO Auto-generated constructor stub

}

PerfectMattMaterial::~PerfectMattMaterial() {
	// TODO Auto-generated destructor stub
}

Ray* PerfectMattMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	// We don't give a shit about any of these peramiters, a perfect matt material just absorbs everything.
	(void)hitLocation;
	(void)angle;
	(void)normal;
	(void)perspective_normal;
	(void)u;
	(void)v;
	(void)w;
	(void)wavelength;
	// Return no ray
	return 0;
}

} /* namespace photonCPU */
