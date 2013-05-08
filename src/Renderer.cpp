/*
 * Renderer.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "Renderer.h"

namespace photonCPU {

Renderer::Renderer(Scene* pScene, int width, int height, float film_distance) {
	mScene = pScene;
	// Add a camera to the scene
	mCameraMat = new CameraMaterial(width, height, 10);
	PlaneObject* plane = new PlaneObject(mCameraMat);
	plane->setPosition(0, 0, film_distance);
	mCameraObject = plane;
	mScene->addObject(mCameraObject);
}

Renderer::~Renderer() {
	// Remove the camera from the scene
	mScene->delObject(mCameraObject);
	// Delete the camera
	delete mCameraObject;
	delete mCameraMat;

}

/**
 * This function
 */
void Renderer::performRender(long long int photons, int argc_mpi, char* argv_mpi[]) {

	// Construct MPI
	int size, rank = 0;
	#ifdef PHOTON_MPI
		MPI::Init( argc_mpi, argv_mpi );

		//MPI_Get_processor_name(hostname,&strlen);
		rank = MPI::COMM_WORLD.Get_rank();
		size = MPI::COMM_WORLD.Get_size();
		printf("Hello, world; from process %d of %d\n", rank, size);

		// Adjust number of photons for MPI
		long long int long_size = (long long int) size;
		photons = photons/long_size;
		if(rank==0)	printf("MPI adjusted to %ld photons per thread", photons);
	#endif /* MPI */

	// Setup rand based on our rank
	std::srand(rank);

	// Render loop
	int i = 0;
	do{
		// Render
		doRenderPass(photons);
		// Pointers
		Image* accImg;
		Image* img = mCameraMat->getImage();
		// Create image to copy into

		#ifndef PHOTON_MPI
			accImg = new Image(img);
		#endif /* If not MPI */
		#ifdef PHOTON_MPI

			// Create MPI handles
			accImg = new Image(img->getWidth(), img->getHeight());
			MPI::Win window_r;
			MPI::Win window_g;
			MPI::Win window_b;
			
			// Construct an MPI Window to copy some data into, one for each colour.
			int size_in_bytes = sizeof(float)*img->getWidth()*img->getHeight();
			window_r = MPI::Win::Create(accImg->imageR, size_in_bytes, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD);
			window_g = MPI::Win::Create(accImg->imageG, size_in_bytes, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD);
			window_b = MPI::Win::Create(accImg->imageB, size_in_bytes, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD);

			// Perform transfer
			window_r.Fence(0);
			window_g.Fence(0);
			window_b.Fence(0);
			window_r.Accumulate(
					img->imageR,
					img->getWidth()*img->getHeight(),
					MPI_FLOAT,
					0,
					0,
					img->getWidth()*img->getHeight(),
					MPI_FLOAT,
					MPI_SUM
			);
			window_g.Accumulate(
					img->imageG,
					img->getWidth()*img->getHeight(),
					MPI_FLOAT,
					0,
					0,
					img->getWidth()*img->getHeight(),
					MPI_FLOAT,
					MPI_SUM
			);
			window_b.Accumulate(
					img->imageB,
					img->getWidth()*img->getHeight(),
					MPI_FLOAT,
					0,
					0,
					img->getWidth()*img->getHeight(),
					MPI_FLOAT,
					MPI_SUM
			);
			window_r.Fence(0);
			window_g.Fence(0);
			window_b.Fence(0);
			window_r.Free();
		#endif /* MPI */
		// Output the image
		if(rank==0) {
			// Construct filename
			char sbuffer[100];
			sprintf(sbuffer, "photons-%d.ppm", i);
			// Output
			accImg->saveToPPMFile(sbuffer);
			delete accImg;
			printf("Image outputed!\n");
		}
		// Incriment number of images taken
		i++;
	}while(false);

	#ifdef PHOTON_MPI
	// Teardown MPI
        MPI::Finalize();	
	#endif /* MPI */
}

void Renderer::doRenderPass(long long int photons) {
	Ray* r;
	const int maxBounce = 50;
	int bounceCount = 0;
	bool loop = true;
	printf("[Begin photon tracing]\n");
	// Fire a number of photons into the scene
	for (long long int i = 0; i < photons; i++) {
		// Pick a random light
		AbstractLight* light = mScene->getRandomLight();
		// Get a random ray
		r = light->getRandomRayFromLight();
		long long int hundreth = (photons / 100);
		if (photons % 100 > 0)
			hundreth++;
		//printf("hundreth %d\n", hundreth);
		if ((i % hundreth) == 0) {
			int progress = (int) (100 * (i / (float) photons));
			printf("    %d\n", progress);
		}
		loop = true;
		bounceCount = 0;
		while (loop) {
			loop = false;
			// Shoot
			RenderObject* obj = mScene->getClosestIntersection(r);
			// Did we hit anything?
			if ((obj != 0) && (bounceCount < maxBounce)) {
				bounceCount++;
				Ray* newr = obj->transmitRay(r);
				delete r;
				if (newr != 0) {
					//printf(" rly");
					r = newr;
					loop = true;
				}
			} else {
				// Nahhhhh
				delete r;
			}
		}
	}
}

} /* namespace photonCPU */
