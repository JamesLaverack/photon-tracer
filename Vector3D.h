/*
 * Vector3D.h
 *
 *  Created on: 11 Feb 2013
 *      Author: james
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

namespace photonCPU {

	class Vector3D {
	public:
		float x;
		float y;
		float z;
		Vector3D();
		virtual ~Vector3D();
		float dotProduct(Vector3D b);
	};

} /* namespace photonCPU */
#endif /* VECTOR3D_H_ */
