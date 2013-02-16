/*
 * Vector3D.cpp
 *
 *  Created on: 11 Feb 2013
 *      Author: james
 */

#include "Vector3D.h"

namespace photonCPU {

///////////////////////////////////
// CONSTRUCTORS & DECONSTRUCTORS //
///////////////////////////////////

Vector3D::Vector3D(float xi, float yi, float zi){
	x = xi;
	y = yi;
	z = zi;
}

Vector3D::Vector3D() {
	Vector3D(0.0f, 0.0f, 0.0f);
}

Vector3D::~Vector3D() {
	// TODO Auto-generated destructor stub
}

///////////////
// OPERATORS //
///////////////

Vector3D & Vector3D::operator+=(const Vector3D &rhs) {
  x += rhs.x;
  y += rhs.y;
  z += rhs.z;
  return *this;
}

// Add this instance's value to other, and return a new instance
// with the result.
const Vector3D Vector3D::operator+(const Vector3D &other) const {
  Vector3D result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
  result += other;            // Use += to add other to the copy.
  return result;              // All done!
}

Vector3D & Vector3D::operator-=(const Vector3D &rhs) {
  x -= rhs.x;
  y -= rhs.y;
  z -= rhs.z;
  return *this;
}

// Add this instance's value to other, and return a new instance
// with the result.
const Vector3D Vector3D::operator-(const Vector3D &other) const {
  Vector3D result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
  result -= other;            // Use += to add other to the copy.
  return result;              // All done!
}

/////////////
// METHODS //
/////////////

float Vector3D::dotProduct(Vector3D* b) {
	return x*b->x+y*b->y+z*b->z;
}

/**
 * Tells the vector to normalise itself. THIS IS DONE IN PLACE.
 */
void Vector3D::normaliseSelf() {
	// Sum
	float sum = x+y+z;
	if(sum!=0){
		x = x/sum;
		y = y/sum;
		z = z/sum;
	}
}

/**
 * Sets this vector to be the same as v.
 * This does NOT modify v or allocate memory.
 */
void Vector3D::setTo(Vector3D* v) {
	x = v->x;
	y = v->y;
	z = v->z;
}

/**
 * Sets this vector to be the same as the given values.
 */
void Vector3D::setTo(float xi, float yi, float zi) {
	x = xi;
	y = yi;
	z = zi;
}

/**
 * Prints to standard out
 */
void Vector3D::print() {
	printf("(%f, %f, %f)\n", x, y, z);
}

} /* namespace photonCPU */

