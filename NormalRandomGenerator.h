/*
 * NormalRandomGenerator.h
 *
 *  Created on: Mar 2, 2013
 *      Author: James Laverack
 */

#ifndef NORMALRANDOMGENERATOR_H_
#define NORMALRANDOMGENERATOR_H_

#include <cmath>
#include <cstdlib>

namespace photonCPU {

class NormalRandomGenerator {
private:
	int use_last;
	float y2;
	float maxValue;
	float minValue;
	bool isInRange(float value);
	float getUncappedRandom(float mean, float sd);
public:
	float randFloat();
	NormalRandomGenerator();
	virtual ~NormalRandomGenerator();
	float getRandom(float mean, float sd);
};

}

#endif /* NORMALRANDOMGENERATOR_H_ */
