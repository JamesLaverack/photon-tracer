/*
 * Image.cpp
 *
 *  Created on: Mar 18, 2013
 *      Author: James Laverack
 */

#include "Image.h"

namespace photonCPU {

Image::Image(int width, int height) {
	imageR = (float*) malloc(height*width*sizeof(float));
	imageG = (float*) malloc(height*width*sizeof(float));
	imageB = (float*) malloc(height*width*sizeof(float));
	this->width = width;
	this->height = height;
	initImage();
}

Image::Image(Image* imageIn) {
	this->width = imageIn->getWidth();
	this->height = imageIn->getHeight();
	imageR = (float*) malloc(height*width*sizeof(float));
	imageG = (float*) malloc(height*width*sizeof(float));
	imageB = (float*) malloc(height*width*sizeof(float));
	for(int i=0;i<width;i++ ) {
			for(int j=0;j<height;j++ ) {
				imageR[index(i, j)] = imageIn->imageR[imageIn->index(i, j)];
				imageG[index(i, j)] = imageIn->imageG[imageIn->index(i, j)];
				imageB[index(i, j)] = imageIn->imageB[imageIn->index(i, j)];
			}
		}
}

Image::~Image() {
	free(imageR);
	free(imageG);
	free(imageB);
}

void Image::initImage() {
	for(int i=0;i<width;i++ ) {
		for(int j=0;j<height;j++ ) {
			imageR[index(i, j)] = 0.0f;
			imageG[index(i, j)] = 0.0f;
			imageB[index(i, j)] = 0.0f;
		}
	}
}

int Image::index(int x, int y) {
	int wat =  (x + width*y);
	return wat;
}

void Image::verify() {
	printf("begin verify\n");
	bool rv = true;
	bool gv = true;
	bool bv = true;
	for(int i=0;i<width;i++) {
		for(int j=0;j<height;j++) {
			// Red
			if(imageR[index(i, j)]!=0){
				if (rv) printf("Nonzero red found\n");
				rv = false;
			}
			// Green
			if(imageG[index(i, j)]!=0){
				if (gv) printf("Nonzero green found\n");
				gv = false;
			}
			// Blue
			if(imageB[index(i, j)]!=0){
				if (bv) printf("Nonzero blue found\n");
				bv = false;
			}
		}
	}
}

int toColourInt(float f, int maxVal) {
	if (f>1) return maxVal;
	return int(f*maxVal);
}

int Image::getHeight() {
	return height;
}

int Image::getWidth() {
	return width;
}

void Image::saveToPPMFile(char* filename) {
    FILE* f;
    /*
    char sbuffer[100];
    sprintf(sbuffer, "photons-%d.ppm", fileid);
    fileid++;
    */
    	// Find highest value
	float biggest = 0;
	for(int i=0;i<width*height;i++) {
	        biggest += imageR[i];
		biggest += imageG[i];
		biggest += imageB[i];
	}
	printf("SUM colour value is %f\n", biggest);
	biggest = biggest/(width*height*3);
	printf("Avg colour value is %f\n", biggest);
	// BUild file
	f = fopen(filename, "w");
	int maxVal = 65535;
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", width, height);
	fprintf(f, "%d\n", maxVal);
	fprintf(f, "\n");
	for(int i=width-1;i>=0;i--){
		for(int j=0;j<height;j++){
			fprintf(f,
					" %d %d %d \n",
					toColourInt(imageR[index(j, i)]/biggest, maxVal),
					toColourInt(imageG[index(j, i)]/biggest, maxVal),
					toColourInt(imageB[index(j, i)]/biggest, maxVal)
			       );
		}
	}
	fclose(f);
}

} /* namespace photonCPU */
