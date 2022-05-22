#ifndef SIMPLE_VIO_INITIALIZE_INITIAL_EX_ROTATION_H_
#define SIMPLE_VIO_INITIALIZE_INITIAL_EX_ROTATION_H_

#include <utility>
#include <vector>

#include "utils/eigen_types.h"

namespace SimpleVIO {

// This class help you to calibrate extrinsic rotation between imu and camera
// when your totally don't konw the extrinsic parameter
class InitialEXRotation {
public:
    InitialEXRotation();

public:
    bool CalibrationExRotation(const std::vector<std::pair<Vec3, Vec3>> &corres,
                               const Qd &delta_q_imu,
                               Mat33 &calib_ric_result);

private:
    std::vector<Mat33> Rc_;
    std::vector<Mat33> Rc_g_;
    std::vector<Mat33> Rimu_;

    Mat33 ric_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_INITIALIZE_INITIAL_EX_ROTATION_H_
