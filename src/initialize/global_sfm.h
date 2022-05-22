#ifndef SIMPLE_VIO_INITIALIZE_GLOBAL_SFM_H_
#define SIMPLE_VIO_INITIALIZE_GLOBAL_SFM_H_

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "optimize/problem.h"

#include "utils/eigen_types.h"

namespace SimpleVIO {

struct SFMFeature {
    bool state = false;
    int id     = -1;

    std::vector<std::pair<int, Vec2>> observations;

    double position[3];
    double depth = -1.0;
};

class GlobalSFM {
public:
    GlobalSFM(Problem::OptimizationMethod opt_method)
        : opt_method_(opt_method) {}

    bool ConstructSFM(const int frame_num,
                      const int start_frame_idx,
                      const Mat33 &relative_R,
                      const Vec3 &relative_T,
                      std::vector<Qd> &q,
                      std::vector<Vec3> &T,
                      std::vector<SFMFeature> &sfm_f,
                      std::map<int, Vec3> &sfm_tracked_points);

private:
    bool SolveFrameByPnP(const int frame_idx,
                         Mat33 &initial_R,
                         Vec3 &initial_P,
                         std::vector<SFMFeature> &sfm_f);

    void TriangulateOnePoint(const Mat34 &pose0,
                             const Mat34 &pose1,
                             const Vec2 &point0,
                             const Vec2 &point1,
                             Vec3 &point_3d);

    void TriangulateTwoFrames(const int frame0,
                              const Mat34 &pose0,
                              const int frame1,
                              const Mat34 &pose1,
                              std::vector<SFMFeature> &sfm_f);

private:
    Problem::OptimizationMethod opt_method_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_INITIALIZE_GLOBAL_SFM_H_
