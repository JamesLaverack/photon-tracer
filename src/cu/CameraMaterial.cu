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
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 

// Object spesific
rtBuffer<float4, 2>              output_buffer;
rtDeclareVariable(float2,  camera_size, , );
rtDeclareVariable(int2,  image_size, , );

RT_PROGRAM void closest_hit() {
	const float pi = 3.141;

	//output_buffer[make_uint2(5, 5)] = make_float4(1, 1, 1, 1);
	rtPrintf("hit at %f, %f\n", texcoord.x, texcoord.y);
	
	// Hit from the right side?
	if(std::abs(std::acos(optix::dot(ray.direction, geometric_normal))) <= pi/2) return;
	rtPrintf("confirmed hit at %f, %f\n", texcoord.x, texcoord.y);

	// Adjusted cords
	int adj_x = std::floor(texcoord.x*image_size.x);
	int adj_y = std::floor(texcoord.y*image_size.x);
	rtPrintf("tex coords at %d, %d\n", adj_x, adj_y);
	// Convert wavelength to colour
	//float3 colour = wavelength_to_colour(ray.wavelength);

	// Record in buffer
	output_buffer[make_uint2(adj_x, adj_y)] = make_float4(1, 1, 1, 1);
}