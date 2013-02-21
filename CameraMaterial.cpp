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
	actualWidth = 100.0f;
	actualHeight = 100.0f;
	imageR = (float*) malloc(height*width*sizeof(float));
	imageG = (float*) malloc(height*width*sizeof(float));
	imageB = (float*) malloc(height*width*sizeof(float));
	vdiff = height/2;
	udiff = width/2;
}

CameraMaterial::~CameraMaterial() {
	free(imageR);
	free(imageG);
	free(imageB);
}

int CameraMaterial::index(int x, int y) {
	int wat =  (x + imageWidth*y);
	return wat;
}

Ray* CameraMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w, float wavelength) {
	//printf("(%f,%f)", hitLocation->x, hitLocation->y);
	//printf("<%f,%f>", u, v);
	// Are we in the renderable bit?
	int tu = u*(imageWidth/actualWidth)+udiff;
	int tv = v*(imageHeight/actualHeight)+vdiff;
	//printf("<%d,%d>", tu, tv;
	if((tu>=0)&&(tu<imageWidth)&&(tv>=0)&&(tv<imageHeight)) {
		//printf(" HIT\n");
		// record
		// make red
		imageR[index(tu, tv)] += 0.01;
		imageG[index(tu, tv)] += 0.01;
		// Ray is finished
		return 0;
	} else {
		//printf(" MISS\n");
	}

	Ray* cont = new Ray();
	cont->setDirection(angle);
	cont->setPosition(hitLocation);
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
					toColourInt(imageR[index(i, j)], maxVal),
					toColourInt(imageG[index(i, j)], maxVal),
					toColourInt(imageB[index(i, j)], maxVal)
			       );
		}
	}
	fclose(f);
}

} /* namespace photonCPU */
