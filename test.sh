#!/bin/zsh
cd bin
output=~/$1
mkdir -p output
echo "Using folder $output"
for i in {1..6}
do
	str=10^$i
	value=`echo $str | bc`
	echo $value
	./photon-tracer-cpu --time --num-photons $value > output.file
	cp output.file $output/$i.file
done
