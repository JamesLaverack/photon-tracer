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
	if( launch_index == follow_photon ) {
		rtPrintf("[%d] debug_number      %d\n", launch_index, debug_id);
		rtPrintf("[%d] parametric t      %f\n", launch_index, t_hit);
		rtPrintf("[%d] depth             %d\n", launch_index, prd_photon.depth);
		rtPrintf("[%d] r (orginal)       %f\n", launch_index, index_of_refraction);
		rtPrintf("[%d] ray origin        <%f, %f, %f>\n", launch_index, ray.origin.x, ray.origin.y, ray.origin.z);
		rtPrintf("[%d] hit point         <%f, %f, %f>\n", launch_index, hitpoint.x, hitpoint.y, hitpoint.z);
		rtPrintf("[%d] Orginal Direction <%f, %f, %f>\n", launch_index, ray.direction.x, ray.direction.y, ray.direction.z);
		rtPrintf("[%d] Normal            <%f, %f, %f>\n", launch_index, geometric_normal.x, geometric_normal.y, geometric_normal.z);
	}
	const float pi = 3.141;
	// Do we absorb this?
	if(prd_photon.depth >= scene_bounce_limit) return;

	float r_index = refractive_index(prd_photon.wavelength);//index_of_refraction;
	// Ugly lens hack
	if(std::abs(hitpoint.z)>hack_lens_depth) {
		r_index = 1;
	}
	if((hitpoint.x)*(hitpoint.x) + (hitpoint.y)*(hitpoint.y) > (hack_lens_radius*hack_lens_radius)) {
		r_index = 1;
	}

	float angle_in = std::acos(optix::dot(ray.direction, geometric_normal));
	float3 axis = optix::normalize( optix::cross( ray.direction, geometric_normal));
	if(angle_in < pi/2) {
                // We are coming from the material out
                r_index = 1/r_index;
        } else {
                // From the outside into the material
                angle_in = pi - angle_in;
        }

	float angle_out = std::asin(std::sin(angle_in)/r_index);
	float theta = std::abs( angle_out - angle_in );

	float4 bounce = optix::make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 1);
	// Do some rotation
	optix::Matrix4x4 rot1 = optix::Matrix4x4::rotate(theta, axis);
	bounce = bounce*rot1;

	// Get needed values
	float3 bounce_direction = optix::normalize( optix::make_float3(bounce) );

	if( launch_index == follow_photon ) {
		rtPrintf("[%d] r (current)       %f\n", launch_index, r_index);
		rtPrintf("[%d] Bounce Direction  <%f, %f, %f>\n", launch_index, bounce_direction.x, bounce_direction.y, bounce_direction.z);
	}

	// Fire new ray!
	optix::Ray new_ray = optix::make_Ray(hitpoint, bounce_direction, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_photon prd_bounce;
	prd_bounce.importance = 1.f;
	prd_bounce.depth = prd_photon.depth+1;
	prd_bounce.wavelength = prd_photon.wavelength;
	rtTrace(top_object, new_ray, prd_bounce);
}
