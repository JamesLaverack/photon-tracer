#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <curand_kernel.h>
#include "NormalRandom.h"
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

// Current Ray & Intersection
rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 


// Object spesific
rtDeclareVariable(float, max_wavelength, , );
rtDeclareVariable(float, min_wavelength, , );
rtDeclareVariable(float, standard_deviation, , );

/** Get a random number between */
__device__ __inline__ float cappedNormalRandom(float mean) {
	float const pi = 3.141;
	float result;
	do {
		result = mean+curand_normal(&states[launch_index])*standard_deviation;
		while(result > pi){
			result -= pi;
		}
		while(result < -pi){
			result += pi;
		}
	} while(std::abs(result)>pi/2);
	return result;
}

RT_PROGRAM void closest_hit() {
	// Do we absorb this?
	if(prd_photon.wavelength>max_wavelength) return;
	if(prd_photon.wavelength<min_wavelength) return;
	if(prd_photon.depth >= scene_bounce_limit) return;

	float3 reverse_normal = (geometric_normal)*-1;
	float reflect_angle = std::acos(optix::dot(ray.direction, reverse_normal));
	// project our incident ray onto the plane defined by
	// < hitLocation, normal > and make sure it's a unit vector, this becomes u.
	float m = optix::dot(geometric_normal, ray.direction);
	float3 u_vec = optix::normalize( (ray.direction)-((geometric_normal)*m) );
	// Calculate v from the cross product of u and normal
	float3 v_vec = optix::normalize( optix::cross( u_vec, geometric_normal) );
	reflect_angle = -reflect_angle;
	// get theta, which is the angle between our bounce and the normal in the u direction.
	// Also get phi, the angle between our bounce and the normal in the v direction.
	float theta = cappedNormalRandom(reflect_angle);
	float phi   = cappedNormalRandom(0);
	// Construct our bounce vector, this is our actual reflection.
	float4 bounce = optix::make_float4( geometric_normal.x, geometric_normal.y, geometric_normal.z, 0);
	// Do some rotation
	optix::Matrix4x4 rot1 = optix::Matrix4x4::rotate(theta, v_vec);
	bounce = bounce*rot1;
	optix::Matrix4x4 rot2 = optix::Matrix4x4::rotate(phi  , u_vec);
	bounce = bounce*rot2;
	// Get needed values
	float3 bounce_direction = optix::normalize( optix::make_float3(bounce.x, bounce.y, bounce.z) );
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	// Fire new ray!
	optix::Ray new_ray = optix::make_Ray(hitpoint, bounce_direction, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_photon prd_bounce;
	prd_bounce.importance = 1.f;
	prd_bounce.depth = prd_photon.depth+1;
	prd_bounce.wavelength = prd_photon.wavelength;
	rtTrace(top_object, new_ray, prd_bounce);
}