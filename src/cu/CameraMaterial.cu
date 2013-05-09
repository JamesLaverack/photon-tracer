#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
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
rtDeclareVariable(int, max_intensity, , );
rtDeclareVariable(float,  camera_gamma, , );

__device__ float factorAdjust(float Color, float Factor){
	if(Color == 0.0){
		return 0;
	}else{
		return max_intensity * pow(Color * Factor, camera_gamma);
	}
}

__device__ void convert(float wavelength, float* r, float* g, float* b){
	float Blue;
	float Green;
	float Red;
	float Factor;
	if(wavelength >= 350 && wavelength <= 439){
		Red	= -(wavelength - 440.0f) / (440.0f - 350.0f);
		Green = 0.0;
		Blue	= 1.0;
	}else if(wavelength >= 440 && wavelength <= 489){
		Red	= 0.0;
		Green = (wavelength - 440.0f) / (490.0f - 440.0f);
		Blue	= 1.0;
	}else if(wavelength >= 490 && wavelength <= 509){
		Red = 0.0;
		Green = 1.0;
		Blue = -(wavelength - 510.0f) / (510.0f - 490.0f);
	}else if(wavelength >= 510 && wavelength <= 579){
		Red = (wavelength - 510.0f) / (580.0f - 510.0f);
		Green = 1.0;
		Blue = 0.0;
	}else if(wavelength >= 580 && wavelength <= 644){
		Red = 1.0;
		Green = -(wavelength - 645.0f) / (645.0f - 580.0f);
		Blue = 0.0;
	}else if(wavelength >= 645 && wavelength <= 780){
		Red = 1.0;
		Green = 0.0;
		Blue = 0.0;
	}else{
		Red = 0.0;
		Green = 0.0;
		Blue = 0.0;
	}
	if(wavelength >= 350 && wavelength <= 419){
		Factor = 0.3 + 0.7*(wavelength - 350.0f) / (420.0f - 350.0f);
	}else if(wavelength >= 420 && wavelength <= 700){
		Factor = 1.0;
	}else if(wavelength >= 701 && wavelength <= 780){
		Factor = 0.3 + 0.7*(780.0f - wavelength) / (780.0f - 700.0f);
	}else{
		Factor = 0.0;
	}
	*r = Red*0.1;//*Factor;
	*g = Green;//*Factor*0.1;
	*b = Blue*0.1;//*Factor;
}

RT_PROGRAM void closest_hit() {
	const float pi = 3.141;

	//output_buffer[make_uint2(5, 5)] = make_float4(1, 1, 1, 1);
	
	// Hit from the right side?
	if(std::abs(std::acos(optix::dot(ray.direction, geometric_normal))) <= pi/2) return;

	// Adjusted cords
	int adj_x = std::floor(texcoord.x*image_size.x);
	int adj_y = std::floor(texcoord.y*image_size.x);
	//if(prd_photon.depth == 0) return; //DEBUG CODE
	//if(prd_photon.depth > 0 ) rtPrintf("hit depth %d\n", prd_photon.depth);
	if(threadIdx.x >= 31 && threadIdx.y >= 15) {
		//rtPrintf("Thread <%d, %d, %d> in block <%d, %d, %d>\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
	}
	//rtPrintf("Block id y %d\n", blockIdx.y);
	
	// Record in buffer
	float r, g, b;
	convert(prd_photon.wavelength, &r, &g, &b);
	uint2 address = make_uint2(adj_x, adj_y);
	atomicAdd(&(output_buffer[address].x), r);
	atomicAdd(&(output_buffer[address].y), g);
	atomicAdd(&(output_buffer[address].z), b);
	atomicAdd(&(output_buffer[address].w), 1);
	//printf("R was %f, is now %f.\n", old_r, output_buffer[address].x);
}
