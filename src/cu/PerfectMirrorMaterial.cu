#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <curand_kernel.h>
#include "PerRay.h"

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  photon_ray_type, , );
rtDeclareVariable(unsigned int,  scene_bounce_limit, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint,      launch_index, rtLaunchIndex, );
rtBuffer<curandState, 1>              states;

// Photon ray datatype
rtDeclareVariable(PerRayData_photon, prd_photon, rtPayload, );
rtDeclareVariable(int, follow_photon, , );

// Current Ray & Intersection
rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

// Object spesific
RT_PROGRAM void closest_hit() {
	// Do we absorb this?
	if(prd_photon.depth >= scene_bounce_limit) return;
	float const pi = 3.141;

	float3 i = ray.direction*-1;
	float m = optix::dot(i, geometric_normal)*2;
	float3 reflect = optix::normalize(geometric_normal*m-i);
	// Get needed values
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	// Fire new ray!
	optix::Ray new_ray = optix::make_Ray(hitpoint, reflect, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_photon prd_bounce;
	prd_bounce.importance = 1.f;
	prd_bounce.depth = prd_photon.depth+1;
	prd_bounce.wavelength = prd_photon.wavelength;
	rtTrace(top_object, new_ray, prd_bounce);
}
