/*
 * WavelengthToRGB.h
 *
 *  Created on: Feb 22, 2013
 *      Author: James Laverack
 */

#ifndef WAVELENGTHTORGB_H_
#define WAVELENGTHTORGB_H_

#include <cmath>

namespace photonCPU {



class WavelengthToRGB {
private:
	int factorAdjust(float Color, float Factor);
public:
	int intensityMax;
	float gamma;
	WavelengthToRGB(int pIntensitymax, float pGamma);
	void convert(float wavelength, float* r, float *g, float* b);
	virtual ~WavelengthToRGB();
};

}

#endif /* WAVELENGTHTORGB_H_ */
