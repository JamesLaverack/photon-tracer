/*
 * AbstractLight.h
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#ifndef ABSTRACTLIGHT_H_
#define ABSTRACTLIGHT_H_

namespace photonCPU {

class AbstractLight {
public:
	AbstractLight();
	virtual ~AbstractLight();
	virtual Ray getRandomRayFromLight() = 0;
};

} /* namespace photonCPU */
#endif /* ABSTRACTLIGHT_H_ */
