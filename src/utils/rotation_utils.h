#ifndef SIMPLE_VIO_UTILS_ROTATION_UTILS_H_
#define SIMPLE_VIO_UTILS_ROTATION_UTILS_H_

#include <cmath>

#include "eigen_types.h"

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

namespace SimpleVIO {

inline Qd DeltaQ(const Vec3 &rvec) {
    return Qd(1.0, 0.5 * rvec(0), 0.5 * rvec(1), 0.5 * rvec(2));
}

inline Mat33 SkewSymmetric(const Vec3 &v) {
    Mat33 ret;
    ret << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;

    return ret;
}

inline Mat44 Qleft(const Qd &q) {
    Mat44 ret;
    ret(0, 0)             = q.w();
    ret.block<1, 3>(0, 1) = -q.vec().transpose();
    ret.block<3, 1>(1, 0) = q.vec();
    ret.block<3, 3>(1, 1) = q.w() * Mat33::Identity() + SkewSymmetric(q.vec());

    return ret;
}

inline Mat44 Qright(const Qd &q) {
    Mat44 ret;
    ret(0, 0)             = q.w();
    ret.block<1, 3>(0, 1) = -q.vec().transpose();
    ret.block<3, 1>(1, 0) = q.vec();
    ret.block<3, 3>(1, 1) = q.w() * Mat33::Identity() - SkewSymmetric(q.vec());

    return ret;
}

inline Vec3 R2ypr(const Mat33 &R) {
    Vec3 n = R.col(0);
    Vec3 o = R.col(1);
    Vec3 a = R.col(2);

    double y = std::atan2(n(1), n(0));
    double p = std::atan2(-n(2), n(0) * std::cos(y) + n(1) * std::sin(y));
    double r = std::atan2(a(0) * std::sin(y) - a(1) * std::cos(y),
                          -o(0) * std::sin(y) + o(1) * std::cos(y));

    Vec3 ypr(3);
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return (ypr * 180.0 / M_PI);
}

inline Mat33 ypr2R(const Vec3 &ypr) {
    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Mat33 Rz;
    Rz << std::cos(y), -std::sin(y), 0.0, std::sin(y), std::cos(y), 0.0, 0.0,
        0.0, 1.0;

    Mat33 Ry;
    Ry << std::cos(p), 0.0, std::sin(p), 0.0, 1.0, 0.0, -std::sin(p), 0.0,
        std::cos(p);

    Mat33 Rx;
    Rx << 1.0, 0.0, 0.0, 0.0, std::cos(r), -std::sin(r), 0.0, std::sin(r),
        std::cos(r);

    return (Rz * Ry * Rx);
}

inline Mat33 g2R(const Vec3 &g) {
    Vec3 ng1 = g.normalized();
    Vec3 ng2{0.0, 0.0, 1.0};

    Mat33 R0 = Qd::FromTwoVectors(ng1, ng2).toRotationMatrix();

    double yaw = R2ypr(R0).x();

    R0 = ypr2R(Vec3{-yaw, 0.0, 0.0}) * R0;
    // R0 = ypr2R(Vec3{-90.0, 0.0, 0.0}) * R0;

    return R0;
}

template<typename T>
inline T NormalizeAngle(const T &angle_degrees) {
    const T two_pi(2.0 * 180.0);
    if (angle_degrees > 0) {
        return (angle_degrees -
                two_pi * std::floor((angle_degrees + T(180.0)) / two_pi));
    } else {
        return (angle_degrees +
                two_pi * std::floor((-angle_degrees + T(180.0)) / two_pi));
    }
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_UTILS_ROTATION_UTILS_H_