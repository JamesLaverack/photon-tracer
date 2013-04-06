/*
 * NormalRandomGenerator.cpp
 *
 *  Created on: Mar 2, 2013
 *      Author: James Laverack
 */

#include "NormalRandomGenerator.h"

namespace photonCPU {

NormalRandomGenerator::NormalRandomGenerator() {
	y2 = 0;
	use_last = 0;

	// Range from -pi/2 to pi/2;
	maxValue = 3.141/2;
	minValue = -maxValue;
}

NormalRandomGenerator::~NormalRandomGenerator() {
	// TODO Auto-generated destructor stub
}

float NormalRandomGenerator::randFloat() {
	float f = (float)(rand()/((float)RAND_MAX));
	//printf("float %f\n", f);
	return f;
}

bool NormalRandomGenerator::isInRange(float value) {
	if(value < minValue) return false;
	if(value > maxValue) return false;
	return true;
}

/**
 * Get a normally distributed random number within our range limitations.
 * If outside of our range then throw it away and try again until we get one.
 * Theoretically, or for stupid values of minValue and maxValue this could
 * take a while to return. This will be especially bad if your mean does not
 * lie between minValue and maxValue. So don't set stupid values for min
 * and max and don't set a stupid mean or this will be slow.
 */
float NormalRandomGenerator::getRandom(float mean, float sd) {
	float result;
	const float pi = 3.141;
	do{
		result = getUncappedRandom(mean, sd);
		while(result > pi){
			result -= pi;
		}
		while(result < -pi){
			result += pi;
		}
	}while(!isInRange(result));
	return result;
}

float NormalRandomGenerator::getUncappedRandom(float mean, float sd) {
	/* mean m, standard deviation s */
	float x1, x2, w, y1;

	if (use_last)		        /* use value from previous call */
	{
		y1 = y2;
		use_last = 0;
	}
	else
	{
		do {
			x1 = 2.0 * randFloat() - 1.0;
			x2 = 2.0 * randFloat() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = std::sqrt( (-2.0 * std::log( w ) ) / w );
		y1 = x1 * w;
		y2 = x2 * w;
		use_last = 1;
	}

	return( mean + y1 * sd );
}

}
