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
	image = (float*) calloc(sizeof(float), height*width);
	vdiff = height/2;
	udiff = width/2;
}

CameraMaterial::~CameraMaterial() {
}

int CameraMaterial::index(int x, int y, int z) {
	return x + imageWidth*y + imageWidth*imageHeight*0;
}

Ray* CameraMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, float u, float v, float w) {
	//printf("(%f,%f)", hitLocation->x, hitLocation->y);
	//printf("<%f,%f>", u, v);
	// Are we in the renderable bit?
	int tu = u+udiff;
	int tv = v+vdiff;
	//printf("<%d,%d>", tu, tv);
	if((tu>=0)&&(tu<imageWidth)&&(tv>=0)&&(tv<imageHeight)) {
		//printf(" HIT\n");
		// record
		// make red
		image[index(tu, tv, 0)] += 0.01;
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
					toColourInt(image[index(i, j, 0)], maxVal),
					0,
					0
			       );
		}
	}
	fclose(f);
}

} /* namespace photonCPU */
