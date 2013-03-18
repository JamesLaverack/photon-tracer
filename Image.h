/*
 * Image.h
 *
 *  Created on: Mar 18, 2013
 *      Author: James Laverack
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace photonCPU {

class Image {
private:
	int width, height;
	void initImage();
public:
	// These are public so we can MPI over them, for example.
	float *imageR;
	float *imageG;
	float *imageB;
	int index(int x, int y);
	void verify();
	void saveToPPMFile(char* filename);
	int getWidth();
	int getHeight();
	Image(int width, int height);
	Image(Image *pImage);
	virtual ~Image();
};

} /* namespace photonCPU */
#endif /* IMAGE_H_ */
