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
using photonCPU::Vector3D;

int main(void) {

	int seed = std::time(NULL);
	std::srand(seed);

	photonCPU::AbstractMaterial* mirror = new photonCPU::PerfectMirrorMaterial();
	photonCPU::PlaneObject* p1 = new photonCPU::PlaneObject(mirror);
	p1->setNormal(0, 1, -1);
	p1->setPosition(0, 0, 15);

	photonCPU::PerfectMattMaterial* matt = new photonCPU::PerfectMattMaterial();
	photonCPU::SphereObject* sphere = new photonCPU::SphereObject(matt);
	sphere->setPosition(0, -1, 2.5f);

	photonCPU::SphereObject* sphere2 = new photonCPU::SphereObject(matt);
	sphere2->setPosition(-1, 0, 2.5f);

	photonCPU::PointLight* light = new photonCPU::PointLight(0, 0, 5);

	photonCPU::Scene* s = new photonCPU::Scene();
	s->addLight(light);
	s->addObject(p1);
	s->addObject(sphere);
	s->addObject(sphere2);

	photonCPU::Renderer* render = new photonCPU::Renderer(s, 1000, 1000);
	int million = 1000000;
	render->doRenderPass(10*million);

	puts("!!!Hello World!!!");
	return EXIT_SUCCESS;
}
