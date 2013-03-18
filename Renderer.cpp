/*
 * Renderer.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "Renderer.h"

namespace photonCPU {

Renderer::Renderer(Scene* pScene, int width, int height) {
	mScene = pScene;
	// Add a camera to the scene
	mCameraMat = new CameraMaterial(width, height);
	PlaneObject* plane = new PlaneObject(mCameraMat);
	plane->setPosition(0, 0, -110);
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
void Renderer::performRender(int photons, int argc_mpi, char* argv_mpi[]) {

	// Construct MPI
	int flag;
	int size, rank;
	MPI_Init( &argc_mpi, &argv_mpi );
	MPI_Initialized(&flag);
	if ( flag != TRUE ) {
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	//MPI_Get_processor_name(hostname,&strlen);
	MPI_Comm_size( MPI_COMM_WORLD, &size );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	printf("Hello, world; from process %d of %d\n", rank, size);


	// Render loop
	int i = 0;
	do{
		// Render
		doRenderPass(photons);
		// Pointers
		float *r_pntr;
		float *g_pntr;
		float *b_pntr;
		Image* accImg;
		Image* img = mCameraMat->getImage();

		if(rank==0) {
			// Create image to copy into
			accImg = new Image(mCameraMat->getImage());
			// Set our pointers
			r_pntr = accImg->imageR;
			g_pntr = accImg->imageG;
			b_pntr = accImg->imageB;
		} else {

			r_pntr = img->imageR;
			g_pntr = img->imageG;
			b_pntr = img->imageB;
		}
		// MPI accumulate
		MPI_Accumulate(
				r_pntr,
				img->getWidth()*img->getHeight(),
				MPI_FLOAT,
				0,
				0,
				img->getWidth()*img->getHeight(),
				MPI_FLOAT,
				MPI_PLUS,
				MPI_COMM_WORLD
		);
		MPI_Accumulate(
				g_pntr,
				img->getWidth()*img->getHeight(),
				MPI_FLOAT,
				0,
				0,
				img->getWidth()*img->getHeight(),
				MPI_FLOAT,
				MPI_PLUS,
				MPI_COMM_WORLD
		);
		MPI_Accumulate(
				b_pntr,
				img->getWidth()*img->getHeight(),
				MPI_FLOAT,
				0,
				0,
				img->getWidth()*img->getHeight(),
				MPI_FLOAT,
				MPI_PLUS,
				MPI_COMM_WORLD
		);
		// Output this
		if(rank==0) {
			// Construct filename
			char sbuffer[100];
			sprintf(sbuffer, "photons-%d.ppm", i);
			// Output
			accImg->saveToPPMFile(sbuffer);
			delete accImg;
		}
		// Incriment number of images taken
		i++;
	}while(true);

	// Teardown MPI

}

void Renderer::doRenderPass(int photons) {
	Ray* r;
	const int maxBounce = 50;
	int bounceCount = 0;
	bool loop = true;
	printf("[Begin photon tracing]\n");
	// Fire a number of photons into the scene
	for (int i = 0; i < photons; i++) {

		//printf("<%d>\n", i);
		// Pick a random light
		AbstractLight* light = mScene->getRandomLight();
		// Get a random ray
		r = light->getRandomRayFromLight();
		int hundreth = (photons / 100);
		if (photons % 100 > 0)
			hundreth++;
		//printf("hundreth %d\n", hundreth);
		if ((i % hundreth) == 0) {
			int progress = (int) (100 * (i / (float) photons));
			printf("    %d", progress);
			// output
			/*			if ( (progress % 5 == 0) && (progress != 0)) {
			 mCameraMat->toPPM();
			 printf(" (image saved)");
			 }	*/
			printf("\n");
		}
		loop = true;
		bounceCount = 0;
		while (loop) {
			loop = false;
			// debug
			//printf("    dir: ");
			//r->getDirection().print();
			//printf("    pos: ");
			//r->getPosition().print();
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
				//printf(" STRIKE\n");
			} else {
				//printf(" MISS\n");
				delete r;
			}
		}
	}
}

} /* namespace photonCPU */
