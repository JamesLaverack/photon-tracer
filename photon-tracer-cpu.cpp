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
using photonCPU::Vector3D;
using photonCPU::PointLight;

int main(void) {

	int seed = std::time(NULL);
	std::srand(seed);

	photonCPU::AbstractMaterial* mirror = new photonCPU::PerfectMirrorMaterial();
	photonCPU::PerfectMattMaterial* matt = new photonCPU::PerfectMattMaterial();
	photonCPU::ColourMaterial* blue  = new photonCPU::ColourMaterial(460.0f);
	photonCPU::ColourMaterial* green = new photonCPU::ColourMaterial(530.0f);

	photonCPU::SphereObject* sphere = new photonCPU::SphereObject(green);
	sphere->setPosition(0, -1, 2.5f);

	photonCPU::SphereObject* sphere2 = new photonCPU::SphereObject(mirror);
	sphere2->setPosition(-1, 0, 2.5f);

	photonCPU::PlaneObject* floor = new photonCPU::PlaneObject(green);
	floor->setNormal(-1, 0, 0);
	floor->setPosition(400, 0, 0);


	photonCPU::PointLight* light = new photonCPU::PointLight(0, 0, 5);

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
	//s->addObject(sphere);
	//s->addObject(sphere2);


	photonCPU::Renderer* render = new photonCPU::Renderer(s, 1000, 1000);
	int million = 1000000;
	render->doRenderPass(100*million);


	puts("!!!Hello World!!!");

	/* Free some memory */

	// Delete container objects
	delete render;
	delete s;

	// Delete objects
	delete sphere;
	delete sphere2;
	delete floor;

	// Delete materials
	delete mirror;
	delete matt;
	delete green;
	delete blue;

	// Delete light
	delete light;

	// Delete lighting rig
	for(int i=0;i<numLights;i++) {
		delete lighting_rig[i];
	}

	return EXIT_SUCCESS;
}
