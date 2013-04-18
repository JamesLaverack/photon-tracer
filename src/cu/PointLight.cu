#include <optix_world.h>
#include "Random.h"
#include "PerRay.h"

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  photon_ray_type, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

// Object spesific
rtDeclareVariable(float3,            location, , );

RT_PROGRAM void light() {
	// Point light, so always emmit from 1 place
	//printf("HURF\n");
	//r->getPosition().print();
	//r->getPosition().print();
	//printf("DURF\n");
	// Randomise direction
	//TODO Frame number?
	unsigned int seed = tea<16>(launch_index.y+launch_index.x, 0);
	float a = rnd(seed);
	float b = rnd(seed);
	float phi = 2*3.141*a;
	float theta = std::acos(2*b-1);
	//printf("theta %f phi %f\n", theta, phi);
	float x, y, z;
	// Rotate<1, 0, 0> by theta around the y axis
	x = (float) std::sin(theta)*std::cos(phi);
	z = (float) std::sin(theta)*std::sin(phi);
	y = (float) std::cos(theta);

	//y = 0;
	//printf("<%f, %f, %f>\n", x, y, z);
	//r->getDirection().print();
	float3 ray_direction = make_float3(x, y, z);
	
	optix::Ray ray = optix::make_Ray(location, ray_direction, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_photon prd;
	prd.importance = 1.f;
	prd.depth = 0;
	prd.seed = seed;
	prd.wavelength = rnd(seed)*400+300;

	rtTrace(top_object, ray, prd);
}