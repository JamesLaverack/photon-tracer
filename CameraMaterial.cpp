/*
 * CameraMaterial.cpp
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#include "CameraMaterial.h"

namespace photonCPU {

CameraMaterial::CameraMaterial(int width, int height) {

	imageWidth = width;
	imageHeight = height;
	actualWidth = 1000.0f;
	actualHeight = 1000.0f;
	imageR = (float*) malloc(height*width*sizeof(float));
	imageG = (float*) malloc(height*width*sizeof(float));
	imageB = (float*) malloc(height*width*sizeof(float));
	vdiff = height/2;
	udiff = width/2;
	derp = true;
	converter = new photonCPU::WavelengthToRGB(1, 1.0f);
	initImage();
	printf("DEBUG vdiff %d, udiff %d\n", vdiff, udiff);
	printf("DEBUG imageWidth %d, imageHeight %d\n", imageWidth, imageHeight);
	printf("DEBUG actualWidth %f, actualHeight %f\n", actualWidth, actualHeight);
}

CameraMaterial::~CameraMaterial() {
	free(imageR);
	free(imageG);
	free(imageB);
	delete converter;
}

void CameraMaterial::initImage() {
	for(int i=0;i<imageWidth;i++ ) {
		for(int j=0;j<imageHeight;j++ ) {
			imageR[index(i, j)] = 0.0f;
			imageG[index(i, j)] = 0.0f;
			imageB[index(i, j)] = 0.0f;
		}
	}
}

int CameraMaterial::index(int x, int y) {
	int wat =  (x + imageWidth*y);
	return wat;
}

Ray* CameraMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w, float wavelength) {
	// We don't use the 3rd texture cordinate
	(void)w;

	//printf("(%f,%f)", hitLocation->x, hitLocation->y);
	//printf("<%f,%f>", u, v);

	if(wavelength>0) {
		//printf("STRIKE");
	}

	// Are we in the renderable bit?
	int tu = u*(imageWidth/actualWidth)+udiff;
	int tv = v*(imageHeight/actualHeight)+vdiff;
	if (derp) {
		printf("DEBUG <%f, %f> (%d, %d)\n", u, v, tu, tv);
	}
	//printf("<%d,%d>", tu, tv;
	if((tu>=0)&&(tu<imageWidth)&&(tv>=0)&&(tv<imageHeight)) {
		// Are we on the correct side of the camera?
		if(std::acos((*angle).dotProduct(normal))>(3.141/2)) {
			// Do some debug testing
			if (derp) {
				printf("set\n");
				derp = false;
			}
			// convert to RGB
			float r, g, b;
			converter->convert(wavelength, &r, &g, &b);

			//printset = false;
			//printf("wavelength %f to colour set %f, %f, %f\n", wavelength, r, g, b);
			imageR[index(tu, tv)] += r*0.01;
			imageG[index(tu, tv)] += g*0.01;
			imageB[index(tu, tv)] += b*0.01;
			// Ray is finished
			return 0;
		}
	} else {
		//printf(" MISS\n");
	}

	Ray* cont = new Ray();
	cont->setDirection(angle);
	cont->setPosition(hitLocation);
	cont->wavelength = wavelength;
	return cont;
}

int toColourInt(float f, int maxVal) {

	if (f>1) return maxVal;
	return int(f*maxVal);
}

void CameraMaterial::verify() {
	printf("begin verify\n");
	bool rv = true;
	bool gv = true;
	bool bv = true;
	for(int i=0;i<imageWidth;i++) {
		for(int j=0;j<imageHeight;j++) {
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

void CameraMaterial::toPPM() {
    FILE* f;
	f = fopen("photons.ppm", "w");
	int maxVal = 65535;
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", imageWidth, imageHeight);
	fprintf(f, "%d\n", maxVal);
	fprintf(f, "\n");
	for(int i=imageWidth-1;i>=0;i--){
		for(int j=0;j<imageHeight;j++){
			fprintf(f,
					" %d %d %d \n",
					toColourInt(imageR[index(j, i)], maxVal),
					toColourInt(imageG[index(j, i)], maxVal),
					toColourInt(imageB[index(j, i)], maxVal)
			       );
		}
	}
	fclose(f);
}

} /* namespace photonCPU */
