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
	context->setEntryPointCount( 1 );
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

void sync_all_threads() {
	/*  Force Thread Synchronization  */
	cudaError err = cudaThreadSynchronize();

	/*  Check for and display Error  */
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "Cuda Sync Error! : %s.\n", cudaGetErrorString( err) );
	}
}

/**
 * This function does what it says on the tin.
 */
void OptiXRenderer::performRender(long long int photons, int argc_mpi, char* argv_mpi[], int width, int height) {

    // Create OptiX context
	optix::Context context = optix::Context::create();
	context->setRayTypeCount( 1 );
	
	// Debug, this will make everything SLOOOOOW
	context->setPrintEnabled(true);
	
	
	int tmp[] = { 0 };
	std::vector<int> v( tmp, tmp+1 ); 
	context->setDevices(v.begin(), v.end());
	
	int num_devices = context->getEnabledDeviceCount();
	printf("Using %d devices:\n", num_devices);
	
	std::vector<int> enabled_devices =  context->getEnabledDevices();
	for(int i=0;i<num_devices;i++) {
		printf("    Device #%d [%s]\n", enabled_devices[i], context->getDeviceName(enabled_devices[i]).c_str());
	}
	// Setup CUrand
	//CUDA_CALL(cudaMalloc((void **)&devStates, photons * sizeof(curandState)));
	
	// CU rand buffer
// 	optix::Buffer rand_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_USER, photons );
// 	rand_buffer->setElementSize( sizeof(curandState));
// 	context["curand_states"]->set(rand_buffer);
	
	// Calculate number of threads
	int threads = 500000;
	unsigned int iterations_on_device = 10;
	
	// Set some scene-wide variables
	context["photon_ray_type"]->setUint( 0u );
	context["scene_bounce_limit"]->setUint( 10u );
	context["scene_epsilon"]->setFloat( 1.e-4f );
	context["iterations"]->setUint(iterations_on_device);

	// Convert our existing scene into an OptiX one
	convertToOptiXScene(context, width, height);

	// Create buffer for random numbers
	optix::Buffer random_buffer = context->createBufferForCUDA( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, threads );
	random_buffer->setElementSize(sizeof(curandState));
	curandState* states_ptr[num_devices];
	
	cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax);
	//cudaSetDevice(0);
	//cudaMalloc((void **)&states_ptr_0, threads * sizeof(curandState));
	for(int i=0;i<num_devices;i++) {
		int device_id = enabled_devices[i];
		long memory_in_bytes = threads * sizeof(curandState);
		printf("Allocating %ld bytes of memory on device #%d for random states.\n", memory_in_bytes, device_id);
		cudaSetDevice(device_id);
		cudaFree(0);
		cudaMalloc((void **)&states_ptr[device_id], memory_in_bytes);
		CUDAWrapper executer;
		executer.curand_setup(10, 100, threads, (void **)&states_ptr[device_id], time(NULL), i);
		printf("    Executing...\n");
		sync_all_threads();
		cudaDeviceSynchronize();
		printf("    Binding to OptiX Buffer...\n");
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaPeekAtLastError()));
		random_buffer->setDevicePointer(device_id, (CUdeviceptr) states_ptr[device_id]);
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaGetLastError()));
	}
	
	// Set as buffer on context
	context["states"]->set(random_buffer);

	// Create Image buffer
	optix::Buffer buffer = context->createBufferForCUDA( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, width, height );
	optix::float4* imgs_ptr[num_devices];
	
	//cudaSetDevice(0);
	//cudaMalloc((void **)&states_ptr_0, threads * sizeof(curandState));
	for(int i=0;i<num_devices;i++) {
		int device_id = enabled_devices[i];
		long memory_in_bytes = width * height * sizeof(optix::float4);
		printf("Allocating %ld bytes of memory on device #%d for image result.\n", memory_in_bytes, device_id);
		cudaSetDevice(device_id);
		cudaFree(0);
		cudaMalloc((void **)&imgs_ptr[device_id], memory_in_bytes);
		CUDAWrapper executer;
		executer.img_setup((void **)&imgs_ptr[device_id], width, height);
		printf("    Executing...\n");
		sync_all_threads();
		cudaDeviceSynchronize();
		printf("    Binding to OptiX Buffer...\n");
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaPeekAtLastError()));
		buffer->setDevicePointer(device_id, (CUdeviceptr) imgs_ptr[device_id]);
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaGetLastError()));
	}
	
	// Set as buffer on context
	context["output_buffer"]->set(buffer);

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

	// Validate
	try{
		context->validate();
	}catch(Exception& e){
		printf("Validate error!\n");
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaPeekAtLastError()));
		printf("    OptiX says : %s\n", e.getErrorString().c_str() );
		return;
	}
	// Compile context
	try{
		context->compile();
	}catch(Exception& e){
		printf("Compile error!\n");
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaPeekAtLastError()));
		printf("    OptiX says : %s\n", e.getErrorString().c_str() );
		return;
	}

	printf("Render With:\n");
	printf("    %d photons.\n", photons);
	printf("    %d threads.\n", threads);
	printf("    %d iterations per thread.\n", iterations_on_device);
	int launches = (photons/threads)/iterations_on_device;
	printf("    %d launches.\n", launches);
	
	// Render
	int current_launch = 0;
	try{
		for(current_launch=0;current_launch<launches;current_launch++) {
			context->launch(0 , threads );
		}
	}catch(Exception& e){
		printf("Launch error on launch #%d!\n", current_launch);
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaPeekAtLastError()));
		printf("    OptiX says : %s\n", e.getErrorString().c_str() );
		return;
	}
	// Error time
	
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
		optix::float4* img = (optix::float4*) malloc(width*height*sizeof(optix::float4));
		cudaMemcpy(img, imgs_ptr[0], width*height*sizeof(optix::float4), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		//accImg->imageG = (float*) (mCameraMatOptiX["output_buffer_g"]->getBuffer()->get());
		//accImg->imageB = (float*) (mCameraMatOptiX["output_buffer_b"]->getBuffer()->get());
		// Output
		saveToPPMFile(sbuffer, img, width, height);
		delete accImg;
		printf("Image outputed!\n");
	}

	#ifdef PHOTON_MPI
		// Teardown MPI
		MPI::Finalize();	
	#endif /* MPI */
}

int OptiXRenderer::index(int x, int y, int width) {
	int wat =  (x + width*y);
	return wat;
}

int OptiXRenderer::toColourInt(float f, int maxVal) {
	if (f>1) return maxVal;
	return int(f*maxVal);
}

void OptiXRenderer::saveToPPMFile(char* filename, optix::float4* image, int width, int height) {
	FILE* f;
	/*
	char sbuffer[100];
	sprintf(sbuffer, "photons-%d.ppm", fileid);
	fileid++;
	*/
	// Find highest value
	float biggest = 0;
	for(int i=0;i<width*height;i++) {
		biggest += image[i].x;
		biggest += image[i].y;
		biggest += image[i].z;
	}
	printf("SUM colour value is %f\n", biggest);
	biggest = biggest/(width*height*3);
	printf("Avg colour value is %f\n", biggest);
	biggest = 1;
	// BUild file
	f = fopen(filename, "w");
	int maxVal = 65535;
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", width, height);
	fprintf(f, "%d\n", maxVal);
	fprintf(f, "\n");
	for(int i=width-1;i>=0;i--){
		for(int j=0;j<height;j++){
			optix::float4 pixel = image[index(i, j, width)];
			fprintf(f,
					" %d %d %d \n",
					toColourInt(pixel.x/biggest, maxVal),
					toColourInt(pixel.y/biggest, maxVal),
					toColourInt(pixel.z/biggest, maxVal)
			       );
		}
	}
	fclose(f);
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
