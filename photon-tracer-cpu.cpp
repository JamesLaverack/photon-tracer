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
#include "PerfectMirrorMaterial.h"
#include "Renderer.h"
#include "PointLight.h"
#include "AbstractMaterial.h"
using photonCPU::Vector3D;

int main(void) {

	int seed = std::time(NULL);
	std::srand(seed);
/*
	photonCPU::AbstractMaterial* mirror = new photonCPU::PerfectMirrorMaterial();
	photonCPU::PlaneObject* p1 = new photonCPU::PlaneObject(mirror);
	p1->setNormal(0, 0, -1);
	p1->setPosition(0, 0, 15);
*/
	photonCPU::PointLight* light = new photonCPU::PointLight(0, 0, 10);

	photonCPU::Scene* s = new photonCPU::Scene();
	s->addLight(light);

	photonCPU::Renderer* render = new photonCPU::Renderer(s, 100, 100);
	render->doRenderPass(1000000);

	puts("!!!Hello World!!!");
	return EXIT_SUCCESS;
}
