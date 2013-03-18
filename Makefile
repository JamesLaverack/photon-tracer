CXXFLAGS =	-O3 -g -Wall -W -Wextra -fmessage-length=0

OBJS =		photon-tracer-cpu.o RadiusMaskMaterial.o TransparantMaterial.o NormalRandomGenerator.o ColourMaterial.o WavelengthToRGB.o Vector3D.o Ray.o RenderObject.o PlaneObject.o Scene.o AbstractLight.o AbstractMaterial.o CameraMaterial.o PointLight.o Renderer.o PerfectMirrorMaterial.o PerfectMattMaterial.o SphereObject.o

LIBS =

TARGET =	photon-tracer-cpu

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

clean-img:
	rm photons-?.ppm

clean-all: clean clean-img
