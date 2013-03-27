/*
 * wavelengthToRGB.cpp
 *
 *  Created on: Feb 22, 2013
 *      Author: James Laverack
 */

#include "WavelengthToRGB.h"

namespace photonCPU {

WavelengthToRGB::WavelengthToRGB(int pMaxIntensity, float pGamma) {
	gamma = pGamma;
	intensityMax = pMaxIntensity;
}

WavelengthToRGB::~WavelengthToRGB() {
}

void WavelengthToRGB::convert(float wavelength, float* r, float* g, float* b){
  float Blue;
  float Green;
  float Red;
  float Factor;
  if(wavelength >= 350 && wavelength <= 439){
   Red	= -(wavelength - 440.0f) / (440.0f - 350.0f);
   Green = 0.0;
   Blue	= 1.0;
  }else if(wavelength >= 440 && wavelength <= 489){
   Red	= 0.0;
   Green = (wavelength - 440.0f) / (490.0f - 440.0f);
   Blue	= 1.0;
  }else if(wavelength >= 490 && wavelength <= 509){
   Red = 0.0;
   Green = 1.0;
   Blue = -(wavelength - 510.0f) / (510.0f - 490.0f);
  }else if(wavelength >= 510 && wavelength <= 579){
   Red = (wavelength - 510.0f) / (580.0f - 510.0f);
   Green = 1.0;
   Blue = 0.0;
  }else if(wavelength >= 580 && wavelength <= 644){
   Red = 1.0;
   Green = -(wavelength - 645.0f) / (645.0f - 580.0f);
   Blue = 0.0;
  }else if(wavelength >= 645 && wavelength <= 780){
   Red = 1.0;
   Green = 0.0;
   Blue = 0.0;
  }else{
   Red = 0.0;
   Green = 0.0;
   Blue = 0.0;
  }
  if(wavelength >= 350 && wavelength <= 419){
   Factor = 0.3 + 0.7*(wavelength - 350.0f) / (420.0f - 350.0f);
  }else if(wavelength >= 420 && wavelength <= 700){
     Factor = 1.0;
  }else if(wavelength >= 701 && wavelength <= 780){
   Factor = 0.3 + 0.7*(780.0f - wavelength) / (780.0f - 700.0f);
  }else{
   Factor = 0.0;
 }
  *r = factorAdjust(Red, Factor);
  *g = factorAdjust(Green, Factor);
  *b = factorAdjust(Blue, Factor);
}

int WavelengthToRGB::factorAdjust(float Color, float Factor){
  if(Color == 0.0){
    return 0;
   }else{
    return (int) round(intensityMax * pow(Color * Factor, gamma));
   }
}

}
