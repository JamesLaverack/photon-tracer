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
#include "AreaLight.h"
using photonCPU::Vector3D;
using photonCPU::PointLight;

int main(int argc, char* argv[]) {
	// Set Varaibles to defaults
	const long long int million = 1000000;
	long long int num_photons = 5000000;
	bool time_run = false;
	float modifier = 0.005f;
	float shift = 100;
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
		} else if (!strcmp(arg, "--shift")) {
			// Set the camera modifier
			i++;
			if(i==argc) {
				printf("Not enough arguments to --shift.\n");
				return(1);
			} else {
				shift = atof(argv[i]);
			}
		} else if (!strcmp(arg, "--time")){
			// Do we time the run?
			time_run = true;
		}
	}

	// Report!
	printf("########################################\n");
	printf("#                                      #\n");
	printf("#        PHOTON TRACER RENDERER        #\n");
	printf("#                                      #\n");
	#ifdef PHOTON_MPI
	printf("#             MPI:      ON             #\n");
	#else
	printf("#             MPI:     OFF             #\n");
	#endif
	printf("#                                      #\n");
	#ifdef PHOTON_OPTIX
	printf("#             OPTIX:    ON             #\n");
	#else
	printf("#             OPTIX:   OFF             #\n");
	#endif
	printf("#                                      #\n");
	printf("########################################\n");
	
	// Report values used
	if (time_run) printf("Timing run.\n");

	// Begin setup
	photonCPU::ColourMaterial* white  = new photonCPU::ColourMaterial(300.0f, 1000.0f);
	photonCPU::ColourMaterial* green = new photonCPU::ColourMaterial(495.0f, 570.0f);
	photonCPU::ColourMaterial* red = new photonCPU::ColourMaterial(630.0f, 740.0f);
	red->std = 6;
	photonCPU::ColourMaterial* blue = new photonCPU::ColourMaterial(450.0f, 475.0f);
	blue->std = 1;
	photonCPU::TransparantMaterial* trans_in = new photonCPU::TransparantMaterial();
	trans_in->debug_id = 4;
	photonCPU::TransparantMaterial* trans_out = new photonCPU::TransparantMaterial();
	trans_out->debug_id = 5;
	photonCPU::TransparantMaterial* trans = new photonCPU::TransparantMaterial();
	photonCPU::RadiusMaskMaterial* mask = new photonCPU::RadiusMaskMaterial();
	photonCPU::RadiusMaskMaterial* mask_apature = new photonCPU::RadiusMaskMaterial();
	mask_apature->radius = shift;

	float R = 140;
	float d = 20;

	trans_in->lens_hack_depth = d;
	trans_out->lens_hack_depth = d;

	float film_distance = 250;//170;//12.5f;
	float lens_shift = film_distance-50;
	float radi = std::sqrt(R*R - (R-d)*(R-d));

	trans_in->lens_hack_radius = radi;
	trans_out->lens_hack_radius = radi;
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

	photonCPU::PlaneObject* front = new photonCPU::PlaneObject(mask);
	front->setNormal(0, 0, 1);
	front->setPosition(0, 0, 0);

	photonCPU::PlaneObject* apature = new photonCPU::PlaneObject(mask_apature);
	apature->setNormal(0, 0, 1);
	apature->setPosition(0, 0, 0);

	// Balls
	photonCPU::SphereObject* spherer = new photonCPU::SphereObject(blue);
	spherer->setPosition(25, -40, 80+lens_shift);
	spherer->radius = 10;

	photonCPU::SphereObject* sphereg = new photonCPU::SphereObject(trans);
	sphereg->setPosition(-25, -40, 80+lens_shift);
	sphereg->radius = 10;

	// YOLO walls

	photonCPU::PlaneObject* floor = new photonCPU::PlaneObject(white);
	floor->setNormal(0, 1, 0);
	floor->setPosition(0, -50, 0+lens_shift);

	photonCPU::PlaneObject* top = new photonCPU::PlaneObject(white);
	top->setNormal(0, -1, 0);
	top->setPosition(0, 50, 0+lens_shift);

	photonCPU::PlaneObject* back = new photonCPU::PlaneObject(white);
	back->setNormal(0, 0, -1);
	back->setPosition(0, 0, 100+lens_shift);

	photonCPU::PlaneObject* right = new photonCPU::PlaneObject(red);
	right->setNormal(-1, 0, 0);
	right->setPosition(50, 0, 0+lens_shift);

	photonCPU::PlaneObject* left = new photonCPU::PlaneObject(green);
	left->setNormal(1, 0, 0);
	left->setPosition(-50, 0, 0+lens_shift);

	// Make an area light
	Vector3D* l_pos    = new Vector3D(-50, 50, 0+lens_shift);
	Vector3D* l_normal = new Vector3D(0, -1, 0);
	Vector3D* l_up     = new Vector3D(0, 0, 1);
	Vector3D* l_right  = new Vector3D(1, 0, 0);
	l_pos->print();
	photonCPU::AreaLight* light = new photonCPU::AreaLight(l_pos, l_normal, l_up, l_right, 100, 100, 3.141/2);

	photonCPU::Scene* s = new photonCPU::Scene();

	s->addLight(light);
       	s->addObject(front);
	s->addObject(sphere);
	s->addObject(sphere2);

	s->addObject(apature);

	s->addObject(floor);
	s->addObject(right);
	s->addObject(left);
	s->addObject(back);
	s->addObject(spherer);
	s->addObject(sphereg);
	// Create our renderer
	photonCPU::OptiXRenderer* render = new photonCPU::OptiXRenderer(s);

	// Perform the render iself, and do some timing
	gettimeofday(&tic, NULL);
	render->performRender(num_photons, argc, argv, 1000, 1000, -film_distance);
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
	delete floor;
	delete back;
	delete front;
	delete right;
	delete left;
	delete top;

	// Delete materials
	delete mask;
	delete mask_apature;
	delete green;
	delete white;
	delete red;
	delete trans_in;
	delete trans_out;

	// Delete light
	delete light;

	return EXIT_SUCCESS;
}
