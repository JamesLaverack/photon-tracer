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
using photonCPU::Vector3D;


int main(void) {

	Vector3D vec;

	vec.x = 100.0f;
	printf("(%f, %f, %f)\n", vec.x, vec.y, vec.z);


	puts("!!!Hello World!!!");
	return EXIT_SUCCESS;
}
