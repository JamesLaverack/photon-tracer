/*
 * Vector3D.h
 *
 *  Created on: 11 Feb 2013
 *      Author: james
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include <stdio.h>

namespace photonCPU {

	class Vector3D {
	public:
		float x;
		float y;
		float z;
		Vector3D();
		Vector3D(float xi, float yi, float zi);
		virtual ~Vector3D();
		float dotProduct(Vector3D b);
		void normaliseSelf();
		void setTo(Vector3D* v);
		void setTo(float xi, float yi, float zi);
		void print();
	};

} /* namespace photonCPU */
#endif /* VECTOR3D_H_ */
