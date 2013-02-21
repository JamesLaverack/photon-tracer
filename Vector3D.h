/*
 * Vector3D.h
 *
 *  Created on: 11 Feb 2013
 *      Author: james
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include <stdio.h>
#include <cmath>

namespace photonCPU {

	class Vector3D {
	public:
		// Public members
		float x;
		float y;
		float z;
		// Constructors & Deconstructors
		Vector3D();
		Vector3D(Vector3D* pVector);
		Vector3D(float xi, float yi, float zi);
		virtual ~Vector3D();
		// Operators

		// Vector/Vector ops
		Vector3D & operator+=(const Vector3D &rhs);
		const Vector3D operator+(const Vector3D &other) const;
		Vector3D & operator-=(const Vector3D &rhs);
		const Vector3D operator-(const Vector3D &other) const;
		Vector3D & operator*=(const Vector3D &rhs);
		const Vector3D operator*(const Vector3D &other) const;

		// Vector/float ops
		Vector3D & operator+=(const float &rhs);
		const Vector3D operator+(const float &other) const;
		Vector3D & operator-=(const float &rhs);
		const Vector3D operator-(const float &other) const;
		Vector3D & operator*=(const float &rhs);
		const Vector3D operator*(const float &other) const;
		// Methods
		float dotProduct(Vector3D* b);
		float magnitude();
		void normaliseSelf();
		void setTo(Vector3D* v);
		void setTo(float xi, float yi, float zi);
		void print();
	};

} /* namespace photonCPU */
#endif /* VECTOR3D_H_ */
