//go        : photon-tracer-cpu.cpp
// Author      : James Laverack
// Version     :
// Copyright   : MIT
// Description : Hello World in C, Ansi-style
//============================================================================

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sys/time.h>
#include "Vector3D.h"
#include "Ray.h"
#include "Scene.h"
#include "PlaneObject.h"
#include "SphereObject.h"
#include "PerfectMirrorMaterial.h"
#include "OptiXRenderer.h"
#include "PointLight.h"
#include "AbstractMaterial.h"
#include "ColourMaterial.h"
#include "TransparantMaterial.h"
#include "RadiusMaskMaterial.h"
using photonCPU::Vector3D;
using photonCPU::PointLight;

int main(int argc, char* argv[]) {
	// Set Varaibles to defaults
	const long long int million = 1000000;
	long long int num_photons = 50*million;
	bool time_run = false;
	float modifier = 0.005f;
	timeval tic, toc;

	// Parse inputs
	for(int i=1;i<argc;i++) {
		char* arg = argv[i];
		// Ugly if-else chain to parse cmd arguments in a dumb way.
		if(!strcmp(arg, "--num-photons")) {
			// Set number of photons
			i++;
			if(i==argc) {
				printf("Not enough arguments to --num-photons.\n");
				return(1);
			} else {
				num_photons = atoi(argv[i])*million;
			}
		} else if (!strcmp(arg, "--camera-mod")) {
			// Set the camera modifier
			i++;
			if(i==argc) {
				printf("Not enough arguments to --camera-mod.\n");
				return(1);
			} else {
				modifier = atof(argv[i]);
			}
		} else if (!strcmp(arg, "--time")){
			// Do we time the run?
			time_run = true;
		}
	}

	// Report values used
	printf("Firing %ld photons.\n", num_photons);
	if (time_run) printf("Timing run.\n");

	// Begin setup
	printf("D.A.N.C.E.\n");

	photonCPU::AbstractMaterial* mirror = new photonCPU::PerfectMirrorMaterial();
	photonCPU::ColourMaterial* white  = new photonCPU::ColourMaterial(300.0f, 1000.0f);
	photonCPU::ColourMaterial* green = new photonCPU::ColourMaterial(495.0f, 570.0f);
	photonCPU::ColourMaterial* red = new photonCPU::ColourMaterial(630.0f, 740.0f);
	red->std = 6;
	photonCPU::ColourMaterial* blue = new photonCPU::ColourMaterial(450.0f, 475.0f);
	photonCPU::TransparantMaterial* trans_in = new photonCPU::TransparantMaterial();
	photonCPU::TransparantMaterial* trans_out = new photonCPU::TransparantMaterial();
	photonCPU::RadiusMaskMaterial* mask = new photonCPU::RadiusMaskMaterial();
	photonCPU::RadiusMaskMaterial* mask_apature = new photonCPU::RadiusMaskMaterial();
	mask_apature->radius = 1;

	float R = 50;
	float d = 20;

	float radi = std::sqrt(R*R - (R-d)*(R-d));
	printf("Apature size %f\n", radi);
	trans_in->radius = radi;
	trans_out->radius = radi;
	mask->radius = radi;

	photonCPU::SphereObject* sphere = new photonCPU::SphereObject(trans_out);
	sphere->setPosition(0, 0, R-d);
	sphere->radius = R;

	photonCPU::SphereObject* sphere2 = new photonCPU::SphereObject(trans_in);
	sphere2->setPosition(0, 0, -R+d);
	sphere2->radius = R;

	// Balls
	photonCPU::SphereObject* spherer = new photonCPU::SphereObject(red);
	spherer->setPosition(25, -40, 120);
	spherer->radius = 10;

	photonCPU::SphereObject* sphereg = new photonCPU::SphereObject(mirror);
	sphereg->setPosition(-25, -40, 60);
	sphereg->radius = 10;

	// YOLO walls

	photonCPU::PlaneObject* floor = new photonCPU::PlaneObject(white);
	floor->setNormal(0, 1, 0);
	floor->setPosition(0, -50, 0);

	photonCPU::PlaneObject* top = new photonCPU::PlaneObject(white);
	top->setNormal(0, -1, 0);
	top->setPosition(0, 50, 0);

	photonCPU::PlaneObject* back = new photonCPU::PlaneObject(white);
	back->setNormal(0, 0, -1);
	back->setPosition(0, 0, 120);

	photonCPU::PlaneObject* front = new photonCPU::PlaneObject(mask);
	front->setNormal(0, 0, 1);
	front->setPosition(0, 0, 0);

	photonCPU::PlaneObject* apature = new photonCPU::PlaneObject(mask_apature);
	apature->setNormal(0, 0, 1);
	apature->setPosition(0, 0, -38);

	photonCPU::PlaneObject* right = new photonCPU::PlaneObject(green);
	right->setNormal(-1, 0, 0);
	right->setPosition(50, 0, 0);

	photonCPU::PlaneObject* left = new photonCPU::PlaneObject(blue);
	left->setNormal(1, 0, 0);
	left->setPosition(-50, 0, 0);

	photonCPU::PointLight* light = new photonCPU::PointLight(0, 45, 50);

	photonCPU::Scene* s = new photonCPU::Scene();

	// Make a lighting rig
	int numLights = 5;
	float radius = 10;
	float height = 20;
	PointLight** lighting_rig = new PointLight*[numLights];
	for(int i=0;i<numLights;i++) {
		float theta = (i/(numLights+1)) * 3.141 * 2;
		lighting_rig[i] = new photonCPU::PointLight(radius*std::cos(theta), height, radius*std::sin(theta));
		//s->addLight(lighting_rig[i]);
	}



	s->addLight(light);

	s->addObject(floor);
	s->addObject(top);
	s->addObject(right);
	s->addObject(left);
	s->addObject(back);

       	s->addObject(front);
       	s->addObject(apature);
	s->addObject(sphere);
	s->addObject(sphere2);
	s->addObject(spherer);
	s->addObject(sphereg);

	// Create our renderer
	photonCPU::OptiXRenderer* render = new photonCPU::OptiXRenderer(s, 1000, 1000, modifier);

	// Perform the render iself, and do some timing
	gettimeofday(&tic, NULL);
	render->performRender(num_photons, argc, argv, 1000, 1000);
	gettimeofday(&toc, NULL);

	// Report how fast we were, perhaps
	if(time_run) printf("Done in %ld seconds.\n", toc.tv_sec-tic.tv_sec);

	/* Free some memory */

	// Delete container objects
	delete render;
	delete s;

	// Delete objects
	delete sphere;
	delete sphere2;
	delete spherer;
	delete sphereg;
	delete floor;
	delete back;
	delete front;
	delete apature;
	delete right;
	delete left;
	delete top;

	// Delete materials
	delete mask;
	delete mask_apature;
	delete mirror;
	delete green;
	delete white;
	delete red;
	delete trans_in;
	delete trans_out;

	// Delete light
	delete light;

	// Delete lighting rig
	for(int i=0;i<numLights;i++) {
		delete lighting_rig[i];
	}

	return EXIT_SUCCESS;
}
