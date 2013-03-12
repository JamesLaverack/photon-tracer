//============================================================================
// Name        : photon-tracer-cpu.cpp
// Author      : James Laverack
// Version     :
// Copyright   : MIT
// Description : Hello World in C, Ansi-style
//============================================================================

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "Vector3D.h"
#include "Ray.h"
#include "Scene.h"
#include "PlaneObject.h"
#include "SphereObject.h"
#include "PerfectMirrorMaterial.h"
#include "PerfectMattMaterial.h"
#include "Renderer.h"
#include "PointLight.h"
#include "AbstractMaterial.h"
#include "ColourMaterial.h"
#include "TransparantMaterial.h"
using photonCPU::Vector3D;
using photonCPU::PointLight;

int main(void) {

	int seed = std::time(NULL);
	std::srand(seed);
	printf("D.A.N.C.E.\n");

	photonCPU::AbstractMaterial* mirror = new photonCPU::PerfectMirrorMaterial();
	photonCPU::PerfectMattMaterial* matt = new photonCPU::PerfectMattMaterial();
	photonCPU::ColourMaterial* white  = new photonCPU::ColourMaterial(300.0f, 1000.0f);
	photonCPU::ColourMaterial* green = new photonCPU::ColourMaterial(495.0f, 570.0f);
	photonCPU::ColourMaterial* red = new photonCPU::ColourMaterial(630.0f, 740.0f);
	photonCPU::TransparantMaterial* trans_in = new photonCPU::TransparantMaterial();
	photonCPU::TransparantMaterial* trans_out = new photonCPU::TransparantMaterial();

	float R = 24;
	float d = 1;

	float radi = std::sqrt(R*R - (R-d)*(R-d));
	trans_in->radius = radi;
	trans_out->radius = radi;

	photonCPU::SphereObject* sphere = new photonCPU::SphereObject(trans_out);
	sphere->setPosition(0, 0, R-d);
	sphere->radius = R;

	photonCPU::SphereObject* sphere2 = new photonCPU::SphereObject(trans_in);
	sphere2->setPosition(0, 0, -R+d);
	sphere2->radius = R;

	// YOLO walls

	photonCPU::PlaneObject* floor = new photonCPU::PlaneObject(white);
	floor->setNormal(0, 1, 0);
	floor->setPosition(0, -50, 0);

	photonCPU::PlaneObject* top = new photonCPU::PlaneObject(white);
	top->setNormal(0, -1, 0);
	top->setPosition(0, 50, 0);

	photonCPU::PlaneObject* back = new photonCPU::PlaneObject(white);
	back->setNormal(0, 0, -1);
	back->setPosition(0, 0, 30);

	photonCPU::PlaneObject* front = new photonCPU::PlaneObject(white);
	front->setNormal(0, 0, 1);
	front->setPosition(0, 0, 0);

	photonCPU::PlaneObject* right = new photonCPU::PlaneObject(red);
	right->setNormal(-1, 0, 0);
	right->setPosition(50, 0, 0);

	photonCPU::PlaneObject* left = new photonCPU::PlaneObject(green);
	left->setNormal(1, 0, 0);
	left->setPosition(-50, 0, 0);

	photonCPU::PointLight* light = new photonCPU::PointLight(0, 0, 50);

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
/*
	s->addObject(floor);
	s->addObject(top);
	s->addObject(right);
	s->addObject(left);
*/

	//s->addObject(back);
//	/s->addObject(front);
	s->addObject(sphere);
	s->addObject(sphere2);


	photonCPU::Renderer* render = new photonCPU::Renderer(s, 1000, 1000);
	int million = 1000000;
	render->doRenderPass(100*million);


	puts("Done!");

	/* Free some memory */

	// Delete container objects
	delete render;
	delete s;

	// Delete objects
	delete sphere;
	delete sphere2;
	delete floor;
	delete back;
	delete front;
	delete right;
	delete left;
	delete top;

	// Delete materials
	delete mirror;
	delete matt;
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
