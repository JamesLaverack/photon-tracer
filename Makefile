CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		photon-tracer-cpu.o Vector3D.o Ray.o RenderObject.o PlaneObject.o Scene.o AbstractLight.o AbstractMaterial.o CameraMaterial.o PointLight.o Renderer.o PerfectMirrorMaterial.o 

LIBS =

TARGET =	photon-tracer-cpu

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
