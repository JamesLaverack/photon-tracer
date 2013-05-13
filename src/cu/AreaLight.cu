#include <optix_world.h>
#include <curand_kernel.h>
#include "PerRay.h"

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  photon_ray_type, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  iterations, , );
rtDeclareVariable(uint,      launch_index, rtLaunchIndex, );

// Object spesific
rtDeclareVariable(float3,            location, , );
rtDeclareVariable(float3,            normal, , );
rtDeclareVariable(float3,            up, , );
rtDeclareVariable(float3,            right, , );
rtDeclareVariable(float,            width, , );
rtDeclareVariable(float,            height, , );
rtDeclareVariable(float,            variance, , );
rtBuffer<curandState, 1>              states;

RT_PROGRAM void light() {
	float3 pos = location;
	pos += up*curand_uniform(&states[launch_index])*height;
	pos += right*curand_uniform(&states[launch_index])*width;

	float4 ray_direction = make_float4(normal);
	float phi = curand_uniform(&states[launch_index])*variance*2 - variance;
	float theta = curand_uniform(&states[launch_index])*variance*2 - variance;
	ray_direction = ray_direction*optix::Matrix4x4::rotate(phi  , up);
	ray_direction = ray_direction*optix::Matrix4x4::rotate(theta, right);

	optix::Ray ray = optix::make_Ray(pos, make_float3(ray_direction), photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);
	PerRayData_photon prd;
	prd.importance = 1.f;
	prd.depth = 0;
	prd.wavelength = curand_uniform(&states[launch_index])*400+300;
	rtTrace(top_object, ray, prd);
}
