/*
 * CameraMaterial.cpp
 *
 *  Created on: 19 Feb 2013
 *      Author: james
 */

#include "CameraMaterial.h"

namespace photonCPU {

CameraMaterial::CameraMaterial(int width, int height, float modifier) {

	imageWidth = width;
	imageHeight = height;
	actualWidth = 100.0f;
	actualHeight = 100.0f;
	vdiff = height/2;
	udiff = width/2;
	derp = true;
	converter = new photonCPU::WavelengthToRGB(1, 1.0f);
	focalLength = 5;
	apatureSize = 3.141/16;
	fileid = 0;
	this->modifier = modifier;
	img = new photonCPU::Image(width, height);
	printf("DEBUG vdiff %d, udiff %d\n", vdiff, udiff);
	printf("DEBUG imageWidth %d, imageHeight %d\n", imageWidth, imageHeight);
	printf("DEBUG actualWidth %f, actualHeight %f\n", actualWidth, actualHeight);
}

CameraMaterial::~CameraMaterial() {
	delete img;
	delete converter;
}

#ifdef PHOTON_OPTIX
optix::Material CameraMaterial::getOptiXMaterial(optix::Context context) {
	optix::Program chp = context->createProgramFromPTXFile( "ptx/CameraMaterial.ptx", "closest_hit" );
	optix::Material mat = context->createMaterial();
	mat->setClosestHitProgram(0, chp);
	mat["camera_size"]->setFloat(actualWidth, actualHeight);
	mat["image_size"]->setInt(imageWidth, imageHeight);
	mat["max_intensity"]->setInt(1);
	mat["gamma"]->setFloat(1.0f);
	printf("A HURF DURF\n");
	return mat;
}
#endif

Ray* CameraMaterial::transmitRay(Vector3D* hitLocation, Vector3D* angle, Vector3D* normal, Vector3D* perspective_normal, float u, float v, float w, float wavelength) {
	// We don't use the 3rd texture cordinate
	(void)w;
	(void)perspective_normal;

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
		if((std::acos((*angle).dotProduct(normal))>(3.141/2))
		//&& (std::acos((*angle).dotProduct(perspective_normal))>(3.141-apatureSize))
		) {
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
			//float modifier = 0.0005;
			img->imageR[img->index(tu, tv)] += r;
			img->imageG[img->index(tu, tv)] += g;
			img->imageB[img->index(tu, tv)] += b;
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

Image *CameraMaterial::getImage() {
	return img;
}

} /* namespace photonCPU */
