/*
 * AbstractMaterial.h
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#ifndef ABSTRACTMATERIAL_H_
#define ABSTRACTMATERIAL_H_

namespace photonCPU {

class AbstractMaterial {
public:
	AbstractMaterial();
	virtual ~AbstractMaterial();
	virtual Ray transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w) = 0;
};

} /* namespace photonCPU */
#endif /* ABSTRACTMATERIAL_H_ */
