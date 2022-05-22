#ifndef SIMPLE_VIO_ESTIMATOR_ESTIMATOR_H_
#define SIMPLE_VIO_ESTIMATOR_ESTIMATOR_H_

#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "feature/feature_manager.h"
#include "feature/feature_tracker.h"
#include "imu/imu_integration.h"
#include "initialize/initial_alignment.h"
#include "initialize/initial_ex_rotation.h"

#include "parameters.h"
#include "utils/eigen_types.h"

namespace SimpleVIO {

enum class SolverFlag { INITIAL = 0, NON_LINEAR = 1 };

enum class MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

class Estimator {
public:
    Estimator() {}

    bool SetParameters(Parameters *params);

    bool Initialize();

    // interface
public:
    void ProcessIMU(const double dt,
                    const Vec3 &linear_acceleration,
                    const Vec3 &angular_velocity);

    bool ProcessImage(
        const double header,
        const std::map<int, std::vector<FeatureObservation>> &image_features);

    void ClearState();

    double GetTd() const {
        return td_;
    }

    SolverFlag GetSolverFlag() const {
        return solver_flag_;
    }

    std::vector<Mat33> &GetRs() {
        return Rs_;
    }

    std::vector<Vec3> &GetPs() {
        return Ps_;
    }

    std::vector<double> &GetHeaders() {
        return headers_;
    }

private:
    bool InitialStructure();

    bool VisualInitialAlign();

    bool RelativePose(Mat33 &relative_R, Vec3 &relative_T, int &l);

    void SlideWindow();

    void SolveOdometry();

    void SlideWindowNew();

    void SlideWindowOld();

    void BackendOptimization();

    void ProblemSolve();

    void MargOldFrame();

    void MargNewFrame();

    void Vector2Double();

    void Double2Vector();

    bool FailureDetection();

private:
    Parameters *params_ = nullptr;

    FeatureManager feature_manager_;

    InitialEXRotation initial_ex_rotation_;

private:
    int estimate_extrinsic_ = 0;

    //////////////// OUR SOLVER ///////////////////
    MatXX Hprior_;
    VecX bprior_;
    VecX errprior_;
    MatXX Jprior_inv_;

    Mat22 project_sqrt_info_;

    //////////////// OUR SOLVER //////////////////
    SolverFlag solver_flag_ = SolverFlag::INITIAL;

    MarginalizationFlag marginalization_flag_ = MarginalizationFlag::MARGIN_OLD;

    double initial_timestamp_ = 0.0;

    MatXX Ap[2], backup_A;
    VecX bp[2], backup_b;

    // number of camera
    std::vector<Mat33> ric_;
    std::vector<Vec3> tic_;

    // window size + 1
    std::vector<Vec3> Ps_;
    std::vector<Vec3> Vs_;
    std::vector<Mat33> Rs_;
    std::vector<Vec3> Bas_;
    std::vector<Vec3> Bgs_;
    double td_ = 0.0;

    // sliding window
    Mat33 back_R0_;
    Mat33 last_R_;
    Mat33 last_R0_;
    Vec3 back_P0_;
    Vec3 last_P_;
    Vec3 last_P0_;

    std::vector<double> headers_;  // timestamps

    Vec3 G_;

    Vec3 acc_0_;
    Vec3 gyr_0_;
    std::vector<std::shared_ptr<IMUIntegration>> pre_integrations_;

    std::vector<std::vector<double>> dt_buf_;
    std::vector<std::vector<Vec3>> linear_acceleration_buf_;
    std::vector<std::vector<Vec3>> angular_velocity_buf_;

    // current process frame index
    int frame_count_ = 0;

    bool has_first_imu_data = false;

    bool failure_occur_ = false;

    // output
    std::vector<Vec3> point_cloud_;
    std::vector<Vec3> margin_cloud_;
    std::vector<Vec3> key_poses_;

    // double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    // double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    // double para_Feature[NUM_OF_F][SIZE_FEATURE];
    // double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    // double para_Retrive_Pose[SIZE_POSE];
    // double para_Td[1][1];
    // double para_Tr[1][1];

    std::vector<std::vector<double>> para_pose_;
    std::vector<std::vector<double>> para_speed_bias_;
    std::vector<std::vector<double>> para_feature_;
    std::vector<std::vector<double>> para_extrinsic_;
    std::vector<std::vector<double>> para_td_;

    std::map<double, ImageIMUFrame> all_image_frame_;

    std::shared_ptr<IMUIntegration> tmp_pre_integration_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_ESTIMATOR_ESTIMATOR_H_
