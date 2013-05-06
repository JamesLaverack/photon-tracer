#!/bin/zsh
cd bin
photons=10000
echo "Using $photons million photons"
output=~/$1
mkdir -p output
echo "Using folder $output"
for i in {270..290}
do
	./photon-tracer-cpu --time --num-photons $photons --shift $i
	convert photons-0.ppm $output/$i.png
done
