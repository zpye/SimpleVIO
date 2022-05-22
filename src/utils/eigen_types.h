#ifndef SIMPLE_VIO_UTILS_EIGEN_TYPES_H_
#define SIMPLE_VIO_UTILS_EIGEN_TYPES_H_

#include <Eigen/Eigen>

namespace SimpleVIO {

using Scalar = Eigen::Matrix<double, 1, 1>;
using Vec2   = Eigen::Vector2d;
using Vec3   = Eigen::Vector3d;
using Vec4   = Eigen::Vector4d;
using Mat22  = Eigen::Matrix2d;
using Mat33  = Eigen::Matrix3d;
using Mat44  = Eigen::Matrix4d;

using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat26 = Eigen::Matrix<double, 2, 6>;
using Mat32 = Eigen::Matrix<double, 3, 2>;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using Mat36 = Eigen::Matrix<double, 3, 6>;

using VecX  = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using MatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using Qd = Eigen::Quaterniond;

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_UTILS_EIGEN_TYPES_H_