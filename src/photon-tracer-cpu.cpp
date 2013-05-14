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
#include "Renderer.h"
#include "PointLight.h"
#include "AbstractMaterial.h"
#include "ColourMaterial.h"
#include "TransparantMaterial.h"
#include "RadiusMaskMaterial.h"
#include "AreaLight.h"
#ifdef PHOTON_OPTIX
#include "OptiXRenderer.h"
#endif
using photonCPU::Vector3D;
using photonCPU::PointLight;

int main(int argc, char* argv[]) {
	// Set Varaibles to defaults
	const long long int million = 1000000;
	long long int num_photons = 5000000;
	bool time_run = false;
	float modifier = 0.005f;
	timeval tic, toc;

	// Some lens stuff
	float R = 91.575f;
	float ct = 3.8;
	float focal_length = 87.64f + ct/2;
	float lens_diam = 25.0f;

	// General purpse input
	float shift = focal_length*2;
	printf("Shift is set to %f.\n", shift);

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
	shift = 150;
	printf("shift is now %f\n", shift);

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
	photonCPU::ColourMaterial* white  = new photonCPU::ColourMaterial(300.0f, 700.0f);
	photonCPU::ColourMaterial* shade_1  = new photonCPU::ColourMaterial(300.0f, 430.0f);
	photonCPU::ColourMaterial* shade_2  = new photonCPU::ColourMaterial(430.0f, 500.0f);
	photonCPU::ColourMaterial* shade_3  = new photonCPU::ColourMaterial(565.0f, 590.0f);
	photonCPU::ColourMaterial* shade_4  = new photonCPU::ColourMaterial(590.0f, 625.0f);
	photonCPU::ColourMaterial* green  = new photonCPU::ColourMaterial(520.0f, 565.0f);
	photonCPU::ColourMaterial* red    = new photonCPU::ColourMaterial(625.0f, 700.0f);
	photonCPU::ColourMaterial* blue   = new photonCPU::ColourMaterial(450.0f, 490.0f);
	photonCPU::TransparantMaterial* trans_in = new photonCPU::TransparantMaterial();
	trans_in->debug_id = 4;
	photonCPU::TransparantMaterial* trans_out = new photonCPU::TransparantMaterial();
	trans_out->debug_id = 5;
	photonCPU::TransparantMaterial* trans = new photonCPU::TransparantMaterial();
	// Values for SF11 glass
	trans->b1 = 1.73759695;
	trans->b2 = 0.313747346;
	trans->b3 = 1.89878101;
	trans->c1 = 0.013188707;
	trans->c2 = 0.0623068142;
	trans->c3 = 155.23629;
	photonCPU::RadiusMaskMaterial* mask = new photonCPU::RadiusMaskMaterial();
	photonCPU::RadiusMaskMaterial* mask_apature = new photonCPU::RadiusMaskMaterial();
	mask_apature->radius = 100;

	float d = ct/2;

	trans_in->lens_hack_depth = d;
	trans_out->lens_hack_depth = d;

	float film_distance = shift;//170;//12.5f;
	float lens_shift = shift-50;
	float radi = std::sqrt(R*R - (R-d)*(R-d));
	printf("Apature size (radius) %f\n, setting to 40.", radi);
	radi = lens_diam/2;

	trans_in->lens_hack_radius = radi;
	trans_out->lens_hack_radius = radi;
	trans_in->radius = radi;
	trans_out->radius = radi;
	mask->radius = radi;

	photonCPU::SphereObject* lens = new photonCPU::SphereObject(trans_out);
	lens->setPosition(0, 0, R-d);
	lens->radius = R;

	photonCPU::SphereObject* lens2 = new photonCPU::SphereObject(trans_in);
	lens2->setPosition(0, 0, -R+d);
	lens2->radius = R;

	photonCPU::PlaneObject* front = new photonCPU::PlaneObject(mask);
	front->setNormal(0, 0, 1);
	front->setPosition(0, 0, d);

	photonCPU::PlaneObject* apature = new photonCPU::PlaneObject(trans);
	apature->setNormal(0, 0, -1);
	apature->up->setTo(0, 1, 0);
	apature->right->setTo(1, 0, 0);
	apature->setPosition(0, 0, -d);

	// Balls
	photonCPU::SphereObject* sphere1 = new photonCPU::SphereObject(shade_1);
	sphere1->setPosition(-25, 25, 80+lens_shift);
	sphere1->radius = 15;

	photonCPU::SphereObject* sphere2 = new photonCPU::SphereObject(shade_2);
	sphere2->setPosition(25, 25, 80+lens_shift);
	sphere2->radius = 15;

	photonCPU::SphereObject* sphere3 = new photonCPU::SphereObject(shade_3);
	sphere3->setPosition(25, -25, 80+lens_shift);
	sphere3->radius = 15;

	photonCPU::SphereObject* sphere4 = new photonCPU::SphereObject(shade_4);
	sphere4->setPosition(-25, -25, 80+lens_shift);
	sphere4->radius = 15;

	// Prisim
	photonCPU::PlaneObject* prisim_left = new photonCPU::PlaneObject(trans);
	prisim_left->setNormal(-0.866, 0.5, 0);
	prisim_left->up->setTo(0.5, 0.866, 0);
	prisim_left->up->normaliseSelf();
	prisim_left->right->setTo(0, 0, 1);
	prisim_left->width  = 40;
	prisim_left->height = 40;
	prisim_left->setPosition(-prisim_left->height/4, 25, 80+lens_shift);

	photonCPU::PlaneObject* prisim_right = new photonCPU::PlaneObject(trans);
	prisim_right->setNormal(0.866, 0.5, 0);
	prisim_right->up->setTo(-0.5, 0.866, 0);
	prisim_right->up->normaliseSelf();
	prisim_right->right->setTo(0, 0, 1);
	prisim_right->width  = 40;
	prisim_right->height = 40;
	prisim_right->setPosition(prisim_right->height/4, 25, 80+lens_shift);
	// YOLO walls

	photonCPU::PlaneObject* floor = new photonCPU::PlaneObject(white);
	floor->setNormal(0, 1, 0);
	floor->up->setTo(0, 0, 1);
	floor->right->setTo(1, 0, 0);
	floor->setPosition(0, -50, 50+lens_shift);

	photonCPU::PlaneObject* top = new photonCPU::PlaneObject(white);
	top->setNormal(0, -1, 0);
	top->up->setTo(0, 0, 1);
	top->up->setTo(1, 0, 0);
	top->setPosition(0, 50, 50+lens_shift);

	photonCPU::PlaneObject* back = new photonCPU::PlaneObject(white);
	back->setNormal(0, 0, -1);
	back->up->setTo(0, 1, 0);
	back->right->setTo(1, 0, 0);
	back->width = 115;
	back->height = 115;
	back->setPosition(0, 0, 100+lens_shift);

	photonCPU::PlaneObject* right = new photonCPU::PlaneObject(green);
	right->setNormal(-1, 0, 0);
	right->up->setTo(0, 1, 0);
	right->right->setTo(0, 0, 1);
	right->setPosition(50, 0, 50+lens_shift);

	photonCPU::PlaneObject* left = new photonCPU::PlaneObject(red);
	left->setNormal(1, 0, 0);
	left->up->setTo(0, 1, 0);
	left->right->setTo(0, 0, 1);
	left->setPosition(-50, 0, 50+lens_shift);

	// Make an area light
	float l_width = 100;
	float l_height = l_width;
	Vector3D* l_pos    = new Vector3D(-l_width/2, 50, 50+lens_shift-l_height/2);
	Vector3D* l_normal = new Vector3D(0, -1, 0);
	l_normal->normaliseSelf();
	Vector3D* l_up     = new Vector3D(0, 0, 1);
	Vector3D* l_right  = new Vector3D(1, 0, 0);
	l_pos->print();
	photonCPU::AreaLight* light = new photonCPU::AreaLight(l_pos, l_normal, l_up, l_right, l_height, l_width, 3.141/2);

	photonCPU::Scene* s = new photonCPU::Scene();

	s->addLight(light);
       	s->addObject(front);
	s->addObject(lens);
	s->addObject(lens2);

	s->addObject(back);
	s->addObject(right);
	s->addObject(left);
	s->addObject(top);
	s->addObject(floor);
	s->addObject(sphere1);
	s->addObject(sphere2);
	s->addObject(sphere3);
	s->addObject(sphere4);

	// Create our renderer
	#ifdef PHOTON_OPTIX
		photonCPU::OptiXRenderer* render = new photonCPU::OptiXRenderer(s);
	#else
		photonCPU::Renderer* render = new photonCPU::Renderer(s, 1000, 1000, -film_distance);
	#endif
	// Perform the render iself, and do some timing
	gettimeofday(&tic, NULL);
	#ifdef PHOTON_OPTIX
		render->performRender(num_photons, argc, argv, 1000, 1000, -film_distance);
	#else
		render->performRender(num_photons, argc, argv);
	#endif
	gettimeofday(&toc, NULL);

	// Report how fast we were, perhaps
	if(time_run) printf("Done in %ld seconds.\n", toc.tv_sec-tic.tv_sec);

	/* Free some memory */

	// Delete container objects
	delete render;
	delete s;

	// Delete objects
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
