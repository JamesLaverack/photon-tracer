#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "PerRay.h"

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  photon_ray_type, , );
rtDeclareVariable(unsigned int,  scene_bounce_limit, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint,      launch_index, rtLaunchIndex, );
rtDeclareVariable(int, follow_photon, , );

// Photon ray datatype
rtDeclareVariable(PerRayData_photon, prd_photon, rtPayload, );

// Current Ray & Intersection
rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );

// Object spesific
rtDeclareVariable(float, index_of_refraction, , );
rtDeclareVariable(float, hack_lens_depth, , );
rtDeclareVariable(float, hack_lens_radius, , );
rtDeclareVariable(int, debug_id, , );
rtDeclareVariable(float3, b_vals, , );
rtDeclareVariable(float3, c_vals, , );

__device__ __inline__ float refractive_index(const float wavelength) {
	// We take in our wavelength in nm, so convert to µm.
	float lambda = wavelength/1000;
        float lambda_2 = lambda*lambda;
	float3 i = (lambda_2*b_vals)/(lambda_2-c_vals);
	return std::sqrt(1 + i.x + i.y + i.z);
}

RT_PROGRAM void closest_hit() {
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	// Report!
	const float pi = 3.141f;
	// Do we absorb this?
	if(prd_photon.depth >= scene_bounce_limit) return;

	float r_index = refractive_index(prd_photon.wavelength);
	// Ugly lens hack
	if(std::abs(hitpoint.z)>hack_lens_depth) {
		r_index = 1.f;
	}
	if((hitpoint.x)*(hitpoint.x) + (hitpoint.y)*(hitpoint.y) > (hack_lens_radius*hack_lens_radius)) {
		r_index = 1.f;
	}
	float angle_in = acosf(optix::dot(ray.direction, geometric_normal));
	float3 axis = optix::normalize( optix::cross( ray.direction, geometric_normal));
	if(angle_in < pi/2.f) {
                // We are coming from the material out
                r_index = 1.0f/r_index;
        } else {
                // From the outside into the material
                angle_in = pi - angle_in;
        }

	float angle_out = asinf(sinf(angle_in)/r_index);
	float theta = abs( angle_out - angle_in );

	float4 bounce = optix::make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 1);
	// Do some rotation
	bounce = bounce*optix::Matrix4x4::rotate(theta, axis);

	// Get needed values
	// Fire new ray!
	optix::Ray new_ray = optix::make_Ray(hitpoint, optix::normalize( optix::make_float3(bounce) ), photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_photon prd_bounce;
	prd_bounce.importance = 1.f;
	prd_bounce.depth = prd_photon.depth+1;
	prd_bounce.wavelength = prd_photon.wavelength;
	rtTrace(top_object, new_ray, prd_bounce);
}
