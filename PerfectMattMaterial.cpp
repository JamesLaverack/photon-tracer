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

Ray* PerfectMirrorMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w) {
	return 0;
}

} /* namespace photonCPU */
