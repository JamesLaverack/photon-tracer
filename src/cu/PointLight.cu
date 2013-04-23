#include <optix_world.h>
#include "Random.h"
#include "PerRay.h"
#include <curand_kernel.h>

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  photon_ray_type, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  iterations, , );
rtDeclareVariable(uint,      launch_index, rtLaunchIndex, );

// Object spesific
rtDeclareVariable(float3,            location, , );
rtBuffer<curandState, 1>              states;

RT_PROGRAM void light() {
	// Point light, so always emmit from 1 place
	//printf("HURF\n");
	//r->getPosition().print();
	//r->getPosition().print();
	//printf("DURF\n");
	// Randomise direction
	//TODO Frame number?
	/* Copy state to local memory for efficiency */
	curandState localState = states[launch_index];

	for(int i=0;i<iterations;i++) {
		float a = curand_uniform(&localState);
		float b = curand_uniform(&localState);
		float phi = 2*3.141*a;
		float theta = std::acos(2*b-1);
// 		rtPrintf("theta %f phi %f\n", theta, phi);
		float x, y, z;
		// Rotate<1, 0, 0> by theta around the y axis
		x = (float) std::sin(theta)*std::cos(phi);
		z = (float) std::sin(theta)*std::sin(phi);
		y = (float) std::cos(theta);

		float3 ray_direction = make_float3(x, y, z);
		
		optix::Ray ray = optix::make_Ray(location, ray_direction, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

		PerRayData_photon prd;
		prd.importance = 1.f;
		prd.depth = 0;
		prd.wavelength = curand_uniform(&localState)*400+300;
		states[launch_index] = localState;
		rtTrace(top_object, ray, prd);
	}
	
}