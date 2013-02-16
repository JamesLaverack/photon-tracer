//============================================================================
// Name        : photon-tracer-cpu.cpp
// Author      : James Laverack
// Version     :
// Copyright   : MIT
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include "Vector3D.h"
#include "Ray.h"
using photonCPU::Vector3D;


int main(void) {

	photonCPU::Ray vec;

	(vec.direction)->x = 100.0f;
	printf("(%f, %f, %f)\n", (vec.direction)->x, (vec.direction)->y, (vec.direction)->z);


	puts("!!!Hello World!!!");
	return EXIT_SUCCESS;
}
