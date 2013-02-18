//============================================================================
// Name        : photon-tracer-cpu.cpp
// Author      : James Laverack
// Version     :
// Copyright   : MIT
// Description : Hello World in C, Ansi-style
//============================================================================

#include <cstdio>
#include <cstdlib>
#include "Vector3D.h"
#include "Ray.h"
#include "Scene.h"
#include "PlaneObject.h"
using photonCPU::Vector3D;

int main(void) {

	photonCPU::Ray r;
	photonCPU::PlaneObject p1;
	photonCPU::PlaneObject p2;

	r.direction->setTo(30, -20, 100);
	r.direction->normaliseSelf();

	r.position->x = 20;
	r.position->y = 30;

	p1.setNormal(0, 0, -1);
	p2.setNormal(0, 0, -1);

	p1.setPosition(0, 0, 15);
	p2.setPosition(0, 0, 30);

	photonCPU::Scene s;

	s.addObject(&p1);
	s.addObject(&p2);

	((s.getClosestIntersection(&r))->getIntersectionPoint(&r)).print();

	puts("!!!Hello World!!!");
	return EXIT_SUCCESS;
}
