#include <optix_world.h>

RT_PROGRAM void miss()
{
	// Do nothing
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	if( code == RT_EXCEPTION_STACK_OVERFLOW ) {
		rtPrintf("### Stack overflow exception! ###\n");
	} else {
		rtPrintf("### Unknown exception! ###\n");
	}
}
