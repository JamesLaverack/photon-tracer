/*
 * OptiXRenderer.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "OptiXRenderer.h"

namespace photonCPU {

OptiXRenderer::OptiXRenderer(Scene* pScene, int width, int height, float modifier) {
	mScene = pScene;

}

OptiXRenderer::~OptiXRenderer() {
	// Remove the camera from the scene
	mScene->delObject(mCameraObject);
	// Delete the camera
	delete mCameraObject;
	delete mCameraMat;

}

void OptiXRenderer::convertToOptiXScene(optix::Context context, int width, int height) {
	// Setup lighting
	// TODO For now, we assume just one light
	context->setRayGenerationProgram( 0, mScene->mLights[0]->getOptiXLight(context) );

	// Exception program
	// context->setExceptionProgram( 0, context->createProgramFromPTXFile( "ptx/utils.ptx", "exception" ) );
	// context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

	// Miss program
	//  context->setMissProgram( 0, context->createProgramFromPTXFile( "ptx/utils.ptx", "miss" ) );
	//  context["bg_color"]->setFloat( 0.3f, 0.1f, 0.2f );

	// Geometry group
	optix::GeometryGroup geometrygroup = context->createGeometryGroup();
	geometrygroup->setChildCount( mScene->mObjects.size() + 1 );
	geometrygroup->setAcceleration( context->createAcceleration("NoAccel","NoAccel") );

	// Add objects
	for(std::vector<RenderObject*>::size_type i = 0; i != mScene->mObjects.size(); i++) {
		optix::GeometryInstance gi = context->createGeometryInstance();
		// TODO we only support 1 material type per object
		gi->setMaterialCount(1);
		gi->setGeometry(   mScene->mObjects[i]->getOptiXGeometry(context));
		gi->setMaterial(0, mScene->mObjects[i]->getOptiXMaterial(context));
		geometrygroup->setChild(i, gi);
	}

	// Create Camera
	// 
	mCameraMat = new CameraMaterial(width, height, 1);
	PlaneObject* plane = new PlaneObject(mCameraMat);
	plane->setPosition(0, 0, -110);
	mCameraObject = plane;
	// Convert to OptiX
	optix::GeometryInstance gi = context->createGeometryInstance();
	gi->setMaterialCount(1);
	gi->setGeometry(   mCameraObject->getOptiXGeometry(context));
	mCameraMatOptiX = mCameraObject->getOptiXMaterial(context);
	mCameraMatOptiX["width"]->setInt(width);
	mCameraMatOptiX["height"]->setInt(height);
	gi->setMaterial(0, mCameraMatOptiX);
	geometrygroup->setChild(mScene->mObjects.size(), gi);

	context["top_object"]->set(geometrygroup);
}

/**
 * This function does what it says on the tin.
 */
void OptiXRenderer::performRender(long long int photons, int argc_mpi, char* argv_mpi[], int width, int height) {

    // Create OptiX context
	optix::Context context = optix::Context::create();
	context->setRayTypeCount( 1 );
	context->setEntryPointCount( 1 );

	// Convert our existing scene into an OptiX one
	convertToOptiXScene(context, width, height);

	// Create Image buffer
	optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, height );
	mCameraMatOptiX["output_buffer_r"]->set(buffer);

	buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, height );
	mCameraMatOptiX["output_buffer_g"]->set(buffer);

	buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, height );
	mCameraMatOptiX["output_buffer_b"]->set(buffer);
	
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

	// Validate
	context->validate();
	context->compile();

	// Render
    context->launch( 0, width, height );
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
		sprintf(sbuffer, "photons-%d.ppm", 0);
		// Construct buffer
		// TODO fix this memory leak we create right here by loosing the referance to the existing buffers
		// TODO do something nicer than this horrible pointer munge
		//accImg->imageR = (float*) (mCameraMatOptiX["output_buffer_r"]->getBuffer()->get());
		//accImg->imageG = (float*) (mCameraMatOptiX["output_buffer_g"]->getBuffer()->get());
		//accImg->imageB = (float*) (mCameraMatOptiX["output_buffer_b"]->getBuffer()->get());
		// Output
		accImg->saveToPPMFile(sbuffer);
		delete accImg;
		printf("Image outputed!\n");
	}

	#ifdef PHOTON_MPI
	// Teardown MPI
        MPI::Finalize();	
	#endif /* MPI */
}

void OptiXRenderer::doRenderPass(long long int photons) {
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
