#ifndef SIMPLE_VIO_IMU_IMU_UTILS_H_
#define SIMPLE_VIO_IMU_IMU_UTILS_H_

namespace SimpleVIO {

class StateOrder {
public:
    static const int O_P  = 0;
    static const int O_R  = 3;
    static const int O_V  = 6;
    static const int O_BA = 9;
    static const int O_BG = 12;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_IMU_IMU_UTILS_H_