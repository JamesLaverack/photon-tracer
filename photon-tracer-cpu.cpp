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
using photonCPU::Vector3D;


int main(void) {

	Vector3D v1(1, 2, 3);
	Vector3D v2;

	v1.print();
	v2.print();

	v2.setTo(&v1);

	v1.print();
	v2.print();

	v1.x = 0.03f;

	v1.print();
	v2.print();

	Vector3D v3 = v1 + v2;
	v3-=3.0f;
	v3.print();

	photonCPU::Ray r;


	puts("!!!Hello World!!!");
	return EXIT_SUCCESS;
}
