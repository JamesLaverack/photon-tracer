#include <curand_kernel.h>

struct PerRayData_photon
{
	float wavelength;
	float importance;
	unsigned int depth;
};
