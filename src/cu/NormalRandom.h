#include "Random.h"

__host__ __device__ float getUncappedNormalRandom(float mean, float sd, unsigned int seed) {
	/* mean m, standard deviation s */
	float x1, x2, w;

	do {
		x1 = 2.0 * rnd(seed) - 1.0;
		x2 = 2.0 * rnd(seed) - 1.0;
		w = x1 * x1 + x2 * x2;
	} while ( w >= 1.0 );

	w = std::sqrt( (-2.0 * std::log( w ) ) / w );

	return( mean + (x1 * w) * sd );
}

__host__ __device__ float getNormalRandom(float mean, float sd, unsigned int seed) {
	return getUncappedNormalRandom(mean, sd, seed);
	float result;
	const float pi = 3.141;
	do{
		result = getUncappedNormalRandom(mean, sd, seed);
		while(result > pi){
			result -= pi;
		}
		while(result < -pi){
			result += pi;
		}
	}while(std::abs(result)>(pi/2));
	return result;
}