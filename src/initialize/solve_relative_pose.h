#ifndef SIMPLE_VIO_INITIALIZE_SOLVE_RELATIVE_POSE_H_
#define SIMPLE_VIO_INITIALIZE_SOLVE_RELATIVE_POSE_H_

#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Eigen>

#include "utils/eigen_types.h"

namespace SimpleVIO {

void InverseRT(Mat33 &R, Vec3 &T);

bool SolveRelativeRT(const std::vector<std::pair<Vec3, Vec3>> &corres,
                     Mat33 &R,
                     Vec3 &T,
                     const int min_inlier_size      = 12,
                     const size_t min_corres_size   = 15,
                     const double ransac_threshold  = 0.3 / 460,
                     const double ransac_confidence = 0.99);

bool SolveRelativeR(const std::vector<std::pair<Vec3, Vec3>> &corres,
                    Mat33 &R,
                    const int min_inlier_size      = 12,
                    const size_t min_corres_size   = 9,
                    const double ransac_threshold  = 1.0,
                    const double ransac_confidence = 0.99);

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_INITIALIZE_SOLVE_RELATIVE_POSE_H_
