/*
 * OptiXRenderer.cpp
 *
 *  Created on: 18 Feb 2013
 *      Author: james
 */

#include "OptiXRenderer.h"

namespace photonCPU {

OptiXRenderer::OptiXRenderer(Scene* pScene) {
	mScene = pScene;
}

OptiXRenderer::~OptiXRenderer() {
	// Remove the camera from the scene
	mScene->delObject(mCameraObject);
	// Delete the camera
	delete mCameraObject;
	delete mCameraMat;

}

void OptiXRenderer::convertToOptiXScene(optix::Context context, int width, int height, float film_location) {
	// Setup lighting
	// TODO For now, we assume just one light
	context->setEntryPointCount( 1 );
	context->setRayGenerationProgram( 0, mScene->mLights[0]->getOptiXLight(context) );

	// Exception program
	context->setExceptionProgram( 0, context->createProgramFromPTXFile( "ptx/PhotonTracer.ptx", "exception" ) );

	// Miss program
	context->setMissProgram( 0, context->createProgramFromPTXFile( "ptx/PhotonTracer.ptx", "miss" ) );

	// Geometry group
	optix::GeometryGroup geometrygroup = context->createGeometryGroup();
	geometrygroup->setChildCount( mScene->mObjects.size() + 1 );
	geometrygroup->setAcceleration( context->createAcceleration("Bvh","Bvh") );

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
	plane->setPosition(0, 0, film_location); //-65
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

int sync_all_threads() {
	/*  Force Thread Synchronization  */
	cudaError err = cudaDeviceSynchronize();
	/*  Check for and display Error  */
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "Cuda Sync Error! : %s.\n", cudaGetErrorString( err) );
		return 1;
	}
	return 0;
}

/** Prints out "Done." and also provides timings */
void done(timeval start_time) {
	timeval end_time;
	gettimeofday(&end_time, NULL);
	int miliseconds = ((end_time.tv_sec*1000000+end_time.tv_usec)-(start_time.tv_sec*1000000+start_time.tv_usec))/1000;
	int seconds = miliseconds/1000;
	int minutes = seconds/60;
	if(miliseconds<1000) {
		printf("Done. [%d ms]\n", miliseconds);
	} else if (seconds<60) {
		printf("Done. [%d seconds and %d ms]\n", seconds, miliseconds-1000*seconds);
	} else {
		printf("Done. [%d minutes and %d seconds]\n", minutes, seconds-60*minutes);
	}
}

/**
 * This function does what it says on the tin.
 */
void OptiXRenderer::performRender(long long int photons, int argc_mpi, char* argv_mpi[], int width, int height, float film_location) {
	// Keep track of time
	timeval tic;

	// Create OptiX context
	optix::Context context = optix::Context::create();
	context->setRayTypeCount( 1 );

	// Debug, this will make everything SLOOOOOW
	context->setPrintEnabled(false);

	// Set some CUDA flags
	cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

	// Set used devices
	int tmp[] = { 1, 0 };
	std::vector<int> v( tmp, tmp+2 );
	context->setDevices(v.begin(), v.end());

	// Report device usage
	int num_devices = context->getEnabledDeviceCount();
	printf("Using %d devices:\n", num_devices);
	std::vector<int> enabled_devices =  context->getEnabledDevices();
	for(int i=0;i<num_devices;i++) {
		printf("    Device #%d [%s]\n", enabled_devices[i], context->getDeviceName(enabled_devices[i]).c_str());
	}

	// Set some OptiX variables
	context->setStackSize(4096);

	// Report OptiX infomation
	int stack_size_in_bytes = context->getStackSize();
	printf("Optix stack size is %d bytes (~%d KB).\n.", stack_size_in_bytes, stack_size_in_bytes/1024);

	// Declare some variables
	int threads = 500000; //20000000;
	unsigned int iterations_on_device = 1;

	// Set some scene-wide variables
	context["photon_ray_type"]->setUint( 0u );
	context["scene_bounce_limit"]->setUint( 10u );
	context["scene_epsilon"]->setFloat( 1.e-4f );
	context["iterations"]->setUint(iterations_on_device);
	context["follow_photon"]->setInt(66752);

	// Convert our existing scene into an OptiX one
	convertToOptiXScene(context, width, height, film_location);

	// Report infomation
	printf("Rendering with:\n");
	printf("    %lld photons.\n", photons);
	printf("    %d threads.\n", threads);
	printf("    %d iterations per thread.\n", iterations_on_device);
	int launches = (photons/threads)/iterations_on_device;
	if(launches*threads*iterations_on_device<photons) {
		launches++;
		printf("    NOTE: You have asked for %lld photons, we are providing %d photons instead.\n", photons, launches*threads*iterations_on_device);
	}
	printf("    %d optix launches.\n", launches);

	// Create buffer for random numbers
	optix::Buffer random_buffer = context->createBufferForCUDA( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, threads );
	random_buffer->setElementSize(sizeof(curandState));
	curandState* states_ptr[num_devices];

	// Intalise
	for(int i=0;i<num_devices;i++) {
		int device_id = enabled_devices[i];
		long memory_in_bytes = threads * sizeof(curandState);
		long memory_in_megabytes = memory_in_bytes/(1024*1024);
		printf("Allocating %ld bytes (~%ld MB) of memory on device #%d for random states...\n", memory_in_bytes, memory_in_megabytes, device_id);
		gettimeofday(&tic, NULL);
		cudaSetDevice(device_id);
		cudaMalloc((void **)&states_ptr[i], memory_in_bytes);
		done(tic);
		CUDAWrapper executer;
		executer.curand_setup(threads, (void **)&states_ptr[i], time(NULL), i);
	}

	// Set as buffer on context
	context["states"]->set(random_buffer);

	// Wait
	printf("Waiting for random states to initalise...\n");
	gettimeofday(&tic, NULL);
	for(int i=0;i<num_devices;i++) {
		cudaSetDevice(enabled_devices[i]);
		sync_all_threads();
	}
	done(tic);

	// Bind to the OptiX buffer
	// We do this here because it cases a syncronise apparently
	for(int i=0;i<num_devices;i++) {
		random_buffer->setDevicePointer(enabled_devices[i], (CUdeviceptr) states_ptr[i]);
	}

	// Create Image buffer
	optix::Buffer buffer = context->createBufferForCUDA( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, width, height );
	optix::float4* imgs_ptr[num_devices];

	//cudaSetDevice(0);
	//cudaMalloc((void **)&states_ptr_0, threads * sizeof(curandState));
	for(int i=0;i<num_devices;i++) {
		int device_id = enabled_devices[i];
		long memory_in_bytes = width * height * sizeof(optix::float4);
		long memory_in_megabytes = memory_in_bytes/(1024*1024);
		printf("Allocating %ld bytes (~%ld MB) of memory on device #%d for image result...\n", memory_in_bytes, memory_in_megabytes, device_id);
		gettimeofday(&tic, NULL);
		cudaSetDevice(device_id);
		cudaMalloc((void **)&imgs_ptr[i], memory_in_bytes);
		done(tic);
		CUDAWrapper executer;
		executer.img_setup((void **)&imgs_ptr[i], width, height);
	}

	// Set as buffer on context
	context["output_buffer"]->set(buffer);

	// Wait for everytyhing to execute
	printf("Waiting for Image data to initalise...\n");
	gettimeofday(&tic, NULL);
	for(int i=0;i<num_devices;i++) {
		cudaSetDevice(enabled_devices[i]);
		sync_all_threads();
	}
	done(tic);

	// Bind to the OptiX buffer
	// We do this here because it cases a syncronise apparently
	for(int i=0;i<num_devices;i++) {
		buffer->setDevicePointer(enabled_devices[i], (CUdeviceptr) imgs_ptr[i]);
	}

	// Construct MPI
	int size, rank = 0;
	#ifndef PHOTON_MPI
	(void)size;
	(void)argc_mpi;
	(void)argv_mpi;
	#endif

	#ifdef PHOTON_MPI
		MPI::Init( argc_mpi, argv_mpi );

		//MPI_Get_processor_name(hostname,&strlen);
		rank = MPI::COMM_WORLD.Get_rank();
		size = MPI::COMM_WORLD.Get_size();
		printf("Hello, world; from process %d of %d\n", rank, size);

		// Adjust number of photons for MPI
		long long int long_size = (long long int) size;
		photons = photons/long_size;
		if(rank==0)	printf("MPI adjusted to %lld photons per thread", photons);
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

	// Render
	int current_launch = 0;
	try{
		printf("Begin render...\n");
		gettimeofday(&tic, NULL);
		for(current_launch=0;current_launch<launches;current_launch++) {
			printf("    ... %f percent\n", 100*((current_launch*1.0f*threads)/photons));
			context->launch(0 , threads );
		}
		done(tic);
	}catch(Exception& e){
		printf("Launch error on launch #%d!\n", current_launch);
		printf("    CUDA says  : %s\n",  cudaGetErrorString(cudaPeekAtLastError()));
		printf("    OptiX says : %s\n", e.getErrorString().c_str() );
		return;
	}

	#ifndef PHOTON_MPI

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
		// This is the collected image data on the host
		optix::float4* img_host_ptr = (optix::float4*) malloc(width*height*sizeof(optix::float4));
		// If we have more than one device we have to accumulate everything back into one buffer
		if(num_devices == 1) {
			img_host_ptr = (optix::float4*) malloc(width*height*sizeof(optix::float4));
			cudaMemcpy(img_host_ptr, imgs_ptr[0], width*height*sizeof(optix::float4), cudaMemcpyDeviceToHost);
		} else {
			printf("We are using %d GPUs, accumulating result...", num_devices);
			gettimeofday(&tic, NULL);
			// Create an accumulate buffer on GPU #0#
// 			int device_id = enabled_devices[0];
// 			cudaSetDevice(device_id);
// 			optix::float4* accumulate_dev_ptr;
// 			cudaMalloc((void **)&accumulate_dev_ptr, width*height*sizeof(optix::float4));
// 			// put the array of memory ptrs on device 0
// 			optix::float4** ptrs_dev_ptr;
// 			cudaMalloc((void **)&ptrs_dev_ptr, num_devices*sizeof(optix::float4*));
// 			// Copy data over
// 			cudaMemcpy(ptrs_dev_ptr, imgs_ptr, num_devices*sizeof(optix::float4*), cudaMemcpyHostToDevice);
// 			CUDAWrapper executer;
// 			executer.img_accumulate((void ***)&ptrs_dev_ptr, (void **)&accumulate_dev_ptr, num_devices, width, height);
// 			cudaMemcpy(img_host_ptr, accumulate_dev_ptr, width*height*sizeof(optix::float4), cudaMemcpyDeviceToHost);
			// Copy everything to host and accumulate here
			optix::float4* host_buffers[num_devices];
			for(int i=0;i<num_devices;i++) {
				host_buffers[i] = (optix::float4*) malloc(width*height*sizeof(optix::float4));
				cudaMemcpy(host_buffers[i], imgs_ptr[i], width*height*sizeof(optix::float4), cudaMemcpyDeviceToHost);
			}
			// Acumulate
			for(int i=0;i<width*height;i++) {
				img_host_ptr[i] = make_float4(0, 0, 0, 0);
				for(int j=0;j<num_devices;j++) {
					img_host_ptr[i].x += host_buffers[j][i].x;
					img_host_ptr[i].y += host_buffers[j][i].y;
					img_host_ptr[i].z += host_buffers[j][i].z;
					img_host_ptr[i].w += host_buffers[j][i].w;
				}
			}
			for(int i=0;i<num_devices;i++) {
				free(host_buffers[i]);
			}
			done(tic);
		}
		printf("Saving Image to %s...\n", sbuffer);
		gettimeofday(&tic, NULL);
		saveToPPMFile(sbuffer, img_host_ptr, width, height);
		free(img_host_ptr);
		done(tic);
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

// r,g,b values are from 0 to 1
// h = [0,360], s = [0,1], v = [0,1]
//		if s == 0, then h = -1 (undefined)

void rgb2hsv( float r, float g, float b, float *h, float *s, float *v )
{
	float min, max, delta;

	min = 1.0f; //MIN( r, g, b );
	if(r<min) min = r;
	if(g<min) min = g;
	if(b<min) min = b;
	max = 0.0f; //MAX( r, g, b );
	if(r>max) max = r;
	if(g>max) max = g;
	if(b>max) max = b;
	*v = max;				// v

	delta = max - min;

	if( max != 0 )
		*s = delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		*s = 0;
		*h = -1;
		return;
	}

	if( r == max )
		*h = ( g - b ) / delta;		// between yellow & magenta
	else if( g == max )
		*h = 2 + ( b - r ) / delta;	// between cyan & yellow
	else
		*h = 4 + ( r - g ) / delta;	// between magenta & cyan

	*h *= 60;				// degrees
	if( *h < 0 )
		*h += 360;

}

void hsv2rgb( float h, float s, float v, float* r, float* g, float* b )
{
	int i;
	float f, p, q, t;

	if( s == 0 ) {
		// achromatic (grey)
		*r = *g = *b = v;
		return;
	}

	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );

	switch( i ) {
		case 0:
			*r = v;
			*g = t;
			*b = p;
			break;
		case 1:
			*r = q;
			*g = v;
			*b = p;
			break;
		case 2:
			*r = p;
			*g = v;
			*b = t;
			break;
		case 3:
			*r = p;
			*g = q;
			*b = v;
			break;
		case 4:
			*r = t;
			*g = p;
			*b = v;
			break;
		default:		// case 5:
			*r = v;
			*g = p;
			*b = q;
			break;
	}

}

float munge(float val) {
	return std::log(val) + 1;
}

void OptiXRenderer::saveToPPMFile(char* filename, optix::float4* image, int width, int height) {
	FILE* f;
	/*
	char sbuffer[100];
	sprintf(sbuffer, "photons-%d.ppm", fileid);
	fileid++;
	*/
	// Find highest value
	float magic_number = 0;
	float biggest = 0;
	for(int i=0;i<width*height;i++) {
		if(image[i].x>biggest) biggest = image[i].x;
		if(image[i].y>biggest) biggest = image[i].y;
		if(image[i].z>biggest) biggest = image[i].z;
	}
	printf("Biggest value is %f\n", biggest);
	float biggest_munge = munge(biggest);
	float max_hits = 0;
	for(int i=0;i<width*height;i++) {
		if(image[i].w>max_hits) max_hits = image[i].w;
	}
	printf("Maximum hits is %d\n", max_hits);
	float sum = 0;
	float sumr, sumg, sumb;
	for(int i=0;i<width*height;i++) {
		sum += image[i].x;
		sumr+= image[i].x;
		sum += image[i].y;
		sumg+= image[i].y;
		sum += image[i].z;
		sumb+= image[i].z;
	}
	printf("SUM colour value is %f\n", sum);
	float average = sum/(width*height*3);
	float avgr = sumr/(width*height);
	float avgg = sumg/(width*height);
	float avgb = sumb/(width*height);
	printf("Avg colour value is %f\n", average);
	// BUild file
	f = fopen(filename, "w");
	int maxVal = 65535;
	fprintf(f, "P3\n");
	fprintf(f, "%d %d\n", width, height);
	fprintf(f, "%d\n", maxVal);
	fprintf(f, "\n");
	for(int i=0;i<width;i++){
		for(int j=height-1;j>=0;j--){
			// Get our pixel
			optix::float4 pixel = image[index(i, j, width)];
			// Do some calculation
			float pixel_high = 0.00000000000000001f;
			if(pixel.x>pixel_high) pixel_high = pixel.x;
			if(pixel.y>pixel_high) pixel_high = pixel.y;
			if(pixel.z>pixel_high) pixel_high = pixel.z;
			//float brightness = 3;//(pixel.w/max_hits)*0.5+0.5;
			float pixel_low = 1000000000000000000000000.0f;
			if(pixel.x<pixel_low) pixel_low = pixel.x;
			if(pixel.y<pixel_low) pixel_low = pixel.y;
			if(pixel.z<pixel_low) pixel_low = pixel.z;
			float avg_pixel_brightness = ((pixel.x+pixel.y+pixel.z)/3);
			// Get RGB values
			float r, g, b; // 0.0f-1.0f
			float  h, s, v; // 0.0f-1.0f
			float  h2, s2, v2; // 0.0f-1.0f
			float mod = avg_pixel_brightness/average;
			r = pixel.x/pixel_high;
			g = pixel.y/pixel_high;
			b = pixel.z/pixel_high;
			// I whip my colours back and forth
			rgb2hsv(r, g, b, &h, &s, &v);
			//v = pixel.w/max_hits;
			//r = pixel.x/pixel_high*mod;
			//g = pixel.y/pixel_high*mod;
			//b = pixel.z/pixel_high*mod;
			//rgb2hsv(r, g, b, &h2, &s2, &v2);
			//if(v>1.0f) v = 1.0f;
			const int factor = 3;
			s = pow(pixel_high - pixel_low, factor)/pow(pixel_high, factor);
			v = munge(pixel.w)/munge(max_hits);
			r = g = b = 0;
			hsv2rgb(h, s, v, &r, &g, &b);
			// Write to file
			fprintf(f," %d %d %d \n",
				toColourInt(r, maxVal),
				toColourInt(g, maxVal),
				toColourInt(b, maxVal)
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
