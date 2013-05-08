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

Vector3D::Vector3D(Vector3D* pVector) {
	x = pVector->x;
	y = pVector->y;
	z = pVector->z;
}

Vector3D::Vector3D() {
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
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

Vector3D & Vector3D::operator*=(const Vector3D &rhs) {
  x *= rhs.x;
  y *= rhs.y;
  z *= rhs.z;
  return *this;
}

// Add this instance's value to other, and return a new instance
// with the result.
const Vector3D Vector3D::operator*(const Vector3D &other) const {
  Vector3D result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
  result *= other;            // Use += to add other to the copy.
  return result;              // All done!
}

Vector3D & Vector3D::operator+=(const float &rhs) {
  x += rhs;
  y += rhs;
  z += rhs;
  return *this;
}

// Add this instance's value to other, and return a new instance
// with the result.
const Vector3D Vector3D::operator+(const float &other) const {
  Vector3D result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
  result += other;            // Use += to add other to the copy.
  return result;              // All done!
}

Vector3D & Vector3D::operator-=(const float &rhs) {
  x -= rhs;
  y -= rhs;
  z -= rhs;
  return *this;
}

// Add this instance's value to other, and return a new instance
// with the result.
const Vector3D Vector3D::operator-(const float &other) const {
  Vector3D result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
  result -= other;            // Use += to add other to the copy.
  return result;              // All done!
}

Vector3D & Vector3D::operator*=(const float &rhs) {
  x *= rhs;
  y *= rhs;
  z *= rhs;
  return *this;
}

// Add this instance's value to other, and return a new instance
// with the result.
const Vector3D Vector3D::operator*(const float &other) const {
  Vector3D result = *this;     // Make a copy of myself.  Same as MyClass result(*this);
  result *= other;            // Use += to add other to the copy.
  return result;              // All done!
}

/////////////
// METHODS //
/////////////

float Vector3D::dotProduct(Vector3D* b) {
	return x*b->x+y*b->y+z*b->z;
}

Vector3D Vector3D::crossProduct(Vector3D vec) {
	Vector3D result;
	/*printf("cross {\n");
	printf("    this <%f, %f, %f>\n", x, y, z);
	printf("    vec  <%f, %f, %f>\n", vec.x, vec.y, vec.z);*/
	result.x = y*vec.z - z*vec.y;
	result.y = z*vec.x - x*vec.z;
	result.z = x*vec.y - y*vec.x;
	/*printf("    res  <%f, %f, %f>\n", result.x, result.y, result.z);
	printf("}\n");*/
	return result;
}

float Vector3D::magnitude() {
	return std::sqrt(x*x+y*y+z*z);
}

/**
 * Rotate this vector around an arbitary axis vector by an amount of angle.
 * Angle should be in radians. We assue that axis is normalised. If it isn't
 * you will fuck this up.
 */
Vector3D Vector3D::rotate(Vector3D *axis, float angle) {
	// We use the matrix
	// |  a  b  c  |
	// |  d  e  f  | = R
	// |  g  h  i  |

	// Precalculate our cos and sin values for our angle
	float cos = std::cos(angle);
	float sin = std::sin(angle);
	// Move out our axis x, y and z values for ease of use
	float x = axis->x;
	float y = axis->y;
	float z = axis->z;

	// Define our rotation matrix members

	float a = cos + x*x*(1-cos);
	float b = x*y*(1-cos) - z*sin;
	float c = x*z*(1-cos) + y*sin;

	float d = y*x*(1-cos) + z*sin;
	float e = cos + y*y*(1-cos);
	float f = y*z*(1-cos) - x*sin;

	float g = z*x*(1-cos) - y*sin;
	float h = x*y*(1-cos) + x*sin;
	float i = cos + z*z*(1-cos);

	// Apply to create our return vector by standard matrix multiplation R*this
	Vector3D result;
	result.x = a*this->x + b*this->y + c*this->z;
	result.y = d*this->x + e*this->y + f*this->z;
	result.z = g*this->x + h*this->y + i*this->z;
	return result;
}

/**
 * Tells the vector to normalise itself. THIS IS DONE IN PLACE.
 */
void Vector3D::normaliseSelf() {
	// Sum
	float sum = magnitude();
	if(sum>0){
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

