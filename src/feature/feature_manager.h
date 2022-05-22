#ifndef SIMPLE_VIO_FEATURE_FEATURE_MANAGER_H_
#define SIMPLE_VIO_FEATURE_FEATURE_MANAGER_H_

#include <map>
#include <memory>
#include <vector>

#include "parameters.h"
#include "utils/eigen_types.h"

namespace SimpleVIO {

struct FeatureObservation {
    int camera_idx = 0;
    Vec3 point;
    Vec2 uv;
    Vec2 velocity;
};

struct FeaturePerFrame {
public:
    FeaturePerFrame(double _timestamp,
                    const FeatureObservation &_feature_observation)
        : timestamp(_timestamp), feature_observation(_feature_observation) {}

public:
    double timestamp;
    FeatureObservation feature_observation;
};

struct TrackedFeaturePerId {
public:
    TrackedFeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame) {}

    int EndFrame() {
        return (int)(start_frame + feature_per_frame_vec.size() - 1);
    }

public:
    int feature_id;
    int start_frame;

public:
    double estimated_depth = -1.0;

    int solve_flag = 0;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    std::vector<FeaturePerFrame> feature_per_frame_vec;
};

class FeatureManager {
public:
    FeatureManager() {}

    bool Initialize(Parameters *params);

    void SetRs(const std::vector<Mat33> &Rs);

    void SetRic(const std::vector<Mat33> &ric);

public:
    void AddFeatures(
        double timestamp,
        int frame_count,
        const std::map<int, std::vector<FeatureObservation>> &features);

    void UpdateFeatures();

    bool CheckParallax(int frame_count);

    void ClearState();

    int GetFeatureCount();

    std::vector<std::shared_ptr<TrackedFeaturePerId>> &GetFeatures();

    VecX GetDepthVector();

    void SetDepth(const VecX &x, bool update_flag = false);

    void ScaleDepth(const double scale);

    std::vector<std::pair<Vec3, Vec3>> GetCorresponding(int frame_count_l,
                                                        int frame_count_r);

    void Triangulate(const std::vector<Vec3> &Ps,
                     const std::vector<Vec3> &tic,
                     const std::vector<Mat33> &ric);

    int GetLastTrackNum();

    void RemoveFailures();

    void RemoveFront(int frame_count);

    void RemoveBack();

    void RemoveBackShiftDepth(const Mat33 &marg_R,
                              const Vec3 &marg_P,
                              const Mat33 &new_R,
                              const Vec3 &new_P);

private:
    double CompensatedParallax2(const TrackedFeaturePerId &f_per_id,
                                int frame_count);

private:
    Parameters *params_ = nullptr;

    std::vector<Mat33> Rs_;
    std::vector<Mat33> ric_;

    int last_track_num_ = 0;
    std::vector<std::shared_ptr<TrackedFeaturePerId>> features_;

    bool need_update_ = false;
    std::vector<int> available_feature_idx_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_FEATURE_FEATURE_MANAGER_H_
