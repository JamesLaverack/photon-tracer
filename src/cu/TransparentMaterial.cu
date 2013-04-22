#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "NormalRandom.h"
#include "PerRay.h"

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  photon_ray_type, , );
rtDeclareVariable(unsigned int,  scene_bounce_limit, , );
rtDeclareVariable(rtObject,      top_object, , );

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

RT_PROGRAM void closest_hit() {
	const float pi = 3.141;
	// Do we absorb this?
	if(prd_photon.depth >= scene_bounce_limit) return;

	float r_index = index_of_refraction;
	// Ugly lens hack
	float3 hitpoint = ray.origin + t_hit * ray.direction;
	if(std::abs(hitpoint.z)>hack_lens_depth) {
			r_index = 1;
	}
	if((hitpoint.x)*(hitpoint.x) + (hitpoint.y)*(hitpoint.y) > (hack_lens_radius*hack_lens_radius)) {
			r_index = 1;
	}

	float3 axis_of_rotation = optix::cross(ray.direction, geometric_normal);

	float angle_in = std::acos(optix::dot(ray.direction, geometric_normal));
	// Which side are we coming from?
	if(angle_in < pi/2) {
		// We are coming from the material out
		r_index = 1/r_index;
	} else {
		// From the outside into the material
		angle_in = pi - angle_in;
	}
	float angle_out = std::asin(std::sin(angle_in)/r_index);
	float theta = angle_in - angle_out;
	// Construct our bounce vector, this is our actual refraction.
	float4 bounce = optix::make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 0);
	// Do some rotation
	optix::Matrix4x4 rot1 = optix::Matrix4x4::rotate(theta, axis_of_rotation);
	bounce = bounce*rot1;

	// Get needed values
	float3 bounce_direction = optix::normalize( optix::make_float3(bounce.x, bounce.y, bounce.z) );
	// Fire new ray!
	optix::Ray new_ray = optix::make_Ray(hitpoint, bounce_direction, photon_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_photon prd_bounce;
	prd_bounce.importance = 1.f;
	prd_bounce.depth = prd_photon.depth+1;
	prd_bounce.rand_state = prd_photon.rand_state;
	prd_bounce.wavelength = prd_photon.wavelength;
	rtTrace(top_object, new_ray, prd_bounce);
}