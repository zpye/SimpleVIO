#include "estimator.h"

#include <sstream>

#include <opencv2/core/eigen.hpp>

#include "backend/edge_imu.h"
#include "backend/edge_reprojection.h"
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_speed_bias.h"

#include "optimize/loss_function.h"
#include "optimize/problem.h"

#include "initialize/global_sfm.h"
#include "initialize/solve_relative_pose.h"

#include "utils/logger.h"
#include "utils/rotation_utils.h"
#include "utils/timer.h"

namespace SimpleVIO {

// set parameters and do initialization
bool Estimator::SetParameters(Parameters *params) {
    if (nullptr == params) {
        LOGE("SimpleVIO", "Empty params !!!");
        return false;
    }

    params_ = params;

    // sliding window
    Ps_.resize(params->window_size + 1);
    Vs_.resize(params->window_size + 1);
    Rs_.resize(params->window_size + 1);
    Bas_.resize(params->window_size + 1);
    Bgs_.resize(params->window_size + 1);

    pre_integrations_.resize(params->window_size + 1);

    headers_.resize(params->window_size + 1);

    dt_buf_.resize(params->window_size + 1);
    linear_acceleration_buf_.resize(params->window_size + 1);
    angular_velocity_buf_.resize(params->window_size + 1);

    // imu-camera extrinsic
    ric_.resize(params->num_of_camera);
    tic_.resize(params->num_of_camera);

    // optimize params
    para_pose_.resize(params->window_size + 1, std::vector<double>(7));
    para_speed_bias_.resize(params->window_size + 1, std::vector<double>(9));
    para_feature_.resize(1000, std::vector<double>(1));
    para_extrinsic_.resize(params->num_of_camera, std::vector<double>(7));
    para_td_.resize(params->num_of_camera, std::vector<double>(1));

    ClearState();

    return true;
}

bool Estimator::Initialize() {
    if (nullptr == params_) {
        LOGE("SimpleVIO", "Empty params !!!");
        return false;
    }

    // imu-camera extrinsic
    estimate_extrinsic_ = params_->estimate_extrinsic;

    for (int i = 0; i < params_->num_of_camera; ++i) {
        ric_[i] = params_->initial_ric[i];
        tic_[i] = params_->initial_tic[i];
    }

    // feature manager
    feature_manager_.Initialize(params_);
    feature_manager_.SetRs(Rs_);
    feature_manager_.SetRic(ric_);

    // information matrix
    project_sqrt_info_ = params_->focal_length / 1.5 * Mat22::Identity();

    // time delay
    td_ = params_->initial_td;

    return true;
}

void Estimator::ClearState() {
    // sliding window
    for (int i = 0; i < params_->window_size + 1; ++i) {
        Rs_[i].setIdentity();
        Ps_[i].setZero();
        Vs_[i].setZero();
        Bas_[i].setZero();
        Bgs_[i].setZero();
        dt_buf_[i].clear();
        linear_acceleration_buf_[i].clear();
        angular_velocity_buf_[i].clear();
        pre_integrations_[i].reset();
    }

    tmp_pre_integration_.reset();

    // imu-camera extrinsic
    for (int i = 0; i < params_->num_of_camera; ++i) {
        ric_[i] = Mat33::Identity();
        tic_[i] = Vec3::Zero();
    }

    G_ = Vec3::Zero();

    // image-imu frame
    for (auto &iter : all_image_frame_) {
        iter.second.pre_integration.reset();
    }
    all_image_frame_.clear();

    // feature manager
    feature_manager_.ClearState();

    solver_flag_ = SolverFlag::INITIAL;

    has_first_imu_data = false;

    frame_count_ = 0;

    initial_timestamp_ = 0.0;

    td_ = params_->initial_td;

    failure_occur_ = false;
}

void Estimator::ProcessIMU(const double dt,
                           const Vec3 &linear_acceleration,
                           const Vec3 &angular_velocity) {
    if (!has_first_imu_data) {
        acc_0_ = linear_acceleration;
        gyr_0_ = angular_velocity;

        has_first_imu_data = true;
    }

    if (!pre_integrations_[frame_count_]) {
        pre_integrations_[frame_count_].reset(
            new IMUIntegration(acc_0_,
                               gyr_0_,
                               Bas_[frame_count_],
                               Bgs_[frame_count_],
                               params_->G,
                               params_->acc_sigma_n,
                               params_->gyr_sigma_n,
                               params_->acc_sigma_w,
                               params_->gyr_sigma_w));
    }

    if (0 != frame_count_) {
        pre_integrations_[frame_count_]->AddMeasure(dt,
                                                    linear_acceleration,
                                                    angular_velocity);

        tmp_pre_integration_->AddMeasure(dt,
                                         linear_acceleration,
                                         angular_velocity);

        dt_buf_[frame_count_].push_back(dt);
        linear_acceleration_buf_[frame_count_].push_back(linear_acceleration);
        angular_velocity_buf_[frame_count_].push_back(angular_velocity);

        // mid-point integration
        Vec3 un_acc_0 = Rs_[frame_count_] * (acc_0_ - Bas_[frame_count_]);
        Vec3 un_gyr   = 0.5 * (gyr_0_ + angular_velocity) - Bgs_[frame_count_];
        Rs_[frame_count_] *= DeltaQ(un_gyr * dt).toRotationMatrix();
        Vec3 un_acc_1 =
            Rs_[frame_count_] * (linear_acceleration - Bas_[frame_count_]);
        Vec3 un_acc = 0.5 * (un_acc_0 + un_acc_1) - G_;
        Ps_[frame_count_] += Vs_[frame_count_] * dt + 0.5 * un_acc * dt * dt;
        Vs_[frame_count_] += un_acc * dt;
    }

    acc_0_ = linear_acceleration;
    gyr_0_ = angular_velocity;
}

bool Estimator::ProcessImage(
    const double header,
    const std::map<int, std::vector<FeatureObservation>> &image_features) {
    // add feature and check parallax
    feature_manager_.AddFeatures(header, frame_count_, image_features);
    if (feature_manager_.CheckParallax(frame_count_)) {
        marginalization_flag_ = MarginalizationFlag::MARGIN_OLD;
    } else {
        marginalization_flag_ = MarginalizationFlag::MARGIN_SECOND_NEW;
    }

    headers_[frame_count_] = header;

    ImageIMUFrame image_imu_frame;
    image_imu_frame.t                    = header;
    image_imu_frame.pre_integration      = tmp_pre_integration_;
    image_imu_frame.feature_observations = image_features;
    all_image_frame_.insert(std::make_pair(header, image_imu_frame));

    tmp_pre_integration_.reset(new IMUIntegration(acc_0_,
                                                  gyr_0_,
                                                  Bas_[frame_count_],
                                                  Bgs_[frame_count_],
                                                  params_->G,
                                                  params_->acc_sigma_n,
                                                  params_->gyr_sigma_n,
                                                  params_->acc_sigma_w,
                                                  params_->gyr_sigma_w));

    if (2 == estimate_extrinsic_) {
        LOGI("SimpleVIO",
             "calibrating extrinsic param, rotation movement is needed");

        if (0 != frame_count_) {
            std::vector<std::pair<Vec3, Vec3>> corres =
                feature_manager_.GetCorresponding(frame_count_ - 1,
                                                  frame_count_);

            Mat33 calib_ric;
            if (initial_ex_rotation_.CalibrationExRotation(
                    corres,
                    pre_integrations_[frame_count_]->GetDeltaQ(),
                    calib_ric)) {
                if (frame_count_ >= params_->window_size) {
                    ric_[0] = calib_ric;

                    params_->initial_ric[0] = calib_ric;

                    estimate_extrinsic_ = 1;
                }
            }
        }
    }

    if (SolverFlag::INITIAL == solver_flag_) {
        if (params_->window_size == frame_count_) {
            bool result = false;
            if (2 != estimate_extrinsic_ &&
                (header - initial_timestamp_) > 0.1) {
                LOGI("SimpleVIO", "Initialization Start!");

                result             = InitialStructure();
                initial_timestamp_ = header;
            }

            if (result) {
                solver_flag_ = SolverFlag::NON_LINEAR;

                SolveOdometry();

                SlideWindow();

                feature_manager_.RemoveFailures();

                last_R_  = Rs_[params_->window_size];
                last_P_  = Ps_[params_->window_size];
                last_R0_ = Rs_[0];
                last_P0_ = Ps_[0];

                LOGI("SimpleVIO", "Initialization Finish!");
            } else {
                SlideWindow();
            }
        } else {
            frame_count_ += 1;
        }
    } else {
        Timer t_solve;

        SolveOdometry();

        LOGD("SimpleVIO", "solver costs: %lf ms", t_solve.End());

        if (FailureDetection()) {
            LOGW("SimpleVIO", "failure detection!");

            failure_occur_ = true;

            ClearState();

            Initialize();

            LOGW("SimpleVIO", "system reboot!");

            return false;
        }

        Timer t_margin;

        SlideWindow();

        feature_manager_.RemoveFailures();

        LOGD("SimpleVIO", "marginalization costs: %lf ms", t_margin.End());

        //  prepare output of VINS
        key_poses_.clear();
        for (int i = 0; i <= params_->window_size; ++i) {
            key_poses_.push_back(Ps_[i]);
        }

        last_R_  = Rs_[params_->window_size];
        last_P_  = Ps_[params_->window_size];
        last_R0_ = Rs_[0];
        last_P0_ = Ps_[0];
    }

    return true;
}

bool Estimator::InitialStructure() {
    Timer t_sfm;

    // check imu observibility
    {
        Vec3 sum_g = Vec3::Zero();

        std::map<double, SimpleVIO::ImageIMUFrame>::iterator frame_it;

        for (frame_it = all_image_frame_.begin(), ++frame_it;
             frame_it != all_image_frame_.end();
             ++frame_it) {
            // accumulate acc
            double dt  = frame_it->second.pre_integration->GetSumDeltaTime();
            Vec3 tmp_g = frame_it->second.pre_integration->GetDeltaV() / dt;
            // LOGD("SimpleVIO",
            //      "tmp_g G (%lf %lf %lf)",
            //      tmp_g(0),
            //      tmp_g(1),
            //      tmp_g(2));
            sum_g += tmp_g;
        }

        Vec3 aver_g = sum_g / (double)(all_image_frame_.size() - 1);
        LOGD("SimpleVIO",
             "aver G (%lf %lf %lf)",
             aver_g(0),
             aver_g(1),
             aver_g(2));

        double var = 0.0;
        for (frame_it = all_image_frame_.begin(), ++frame_it;
             frame_it != all_image_frame_.end();
             ++frame_it) {
            double dt  = frame_it->second.pre_integration->GetSumDeltaTime();
            Vec3 tmp_g = frame_it->second.pre_integration->GetDeltaV() / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }

        // standard deviation
        var = sqrt(var / (double)(all_image_frame_.size() - 1));
        LOGI("SimpleVIO", "IMU standard deviation [%lf]", var);

        if (var < 0.25) {
            LOGW("SimpleVIO", "IMU excitation not enouth!");
            // return false;
        }
    }

    // global sfm
    std::vector<Qd> Q(frame_count_ + 1);
    std::vector<Vec3> T(frame_count_ + 1);

    std::vector<SFMFeature> sfm_f;

    const auto &features = feature_manager_.GetFeatures();
    for (auto f_per_id : features) {
        int imu_j = f_per_id->start_frame;

        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id    = f_per_id->feature_id;

        for (auto f_per_frame : f_per_id->feature_per_frame_vec) {
            Vec3 &pts_j = f_per_frame.feature_observation.point;
            tmp_feature.observations.push_back(
                std::make_pair(imu_j, Vec2(pts_j.x(), pts_j.y())));

            imu_j += 1;
        }

        sfm_f.push_back(tmp_feature);
    }

    LOGD("SimpleVIO", "Compute Relative Pose");

    Mat33 relative_R;
    Vec3 relative_T;
    int start_frame_idx;
    if (!RelativePose(relative_R, relative_T, start_frame_idx)) {
        LOGW("SimpleVIO",
             "Not enough features or parallax; Move device around");

        return false;
    }

    InverseRT(relative_R, relative_T);

    LOGD("SimpleVIO", "Global SFM");

    std::map<int, Vec3> sfm_tracked_points;
    GlobalSFM sfm((Problem::OptimizationMethod)params_->opt_method);
    if (!sfm.ConstructSFM(frame_count_ + 1,
                          start_frame_idx,
                          relative_R,
                          relative_T,
                          Q,
                          T,
                          sfm_f,
                          sfm_tracked_points)) {
        LOGW("SimpleVIO", "global SFM failed!");

        marginalization_flag_ = MarginalizationFlag::MARGIN_OLD;

        return false;
    }

    LOGD("SimpleVIO", "Solve pnp for all frame");

    // solve pnp for all frame
    std::map<double, ImageIMUFrame>::iterator frame_it;

    int i = 0;
    for (frame_it = all_image_frame_.begin();
         frame_it != all_image_frame_.end();
         ++frame_it) {
        Mat33 Ri = Q[i].toRotationMatrix();

        // provide initial guess
        if (frame_it->first == headers_[i]) {
            frame_it->second.is_key_frame = true;

            frame_it->second.R = Ri * ric_[0].transpose();
            frame_it->second.T = T[i];

            Qd R(frame_it->second.R);
            // LOGD("SimpleVIO",
            //      "%d initial R(%lf %lf %lf %lf), T(%lf %lf %lf)",
            //      i,
            //      R.x(),
            //      R.y(),
            //      R.z(),
            //      R.w(),
            //      frame_it->second.T.x(),
            //      frame_it->second.T.y(),
            //      frame_it->second.T.z());

            i += 1;

            continue;
        }

        if (frame_it->first > headers_[i]) {
            // skip
            i += 1;
        }

        Mat33 inital_R = (Q[i].inverse()).toRotationMatrix();
        Vec3 inital_P  = -inital_R * T[i];

        cv::Mat tmp_r;
        cv::eigen2cv(inital_R, tmp_r);

        cv::Mat rvec;
        cv::Rodrigues(tmp_r, rvec);

        cv::Mat t;
        cv::eigen2cv(inital_P, t);

        frame_it->second.is_key_frame = false;

        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.feature_observations) {
            for (auto &i_p : id_pts.second) {
                auto iter = sfm_tracked_points.find(id_pts.first);
                if (sfm_tracked_points.end() != iter) {
                    Vec3 world_pts = iter->second;
                    pts_3_vector.push_back(
                        cv::Point3f(world_pts(0), world_pts(1), world_pts(2)));

                    Vec2 img_pts = i_p.point.head<2>();
                    pts_2_vector.push_back(cv::Point2f(img_pts(0), img_pts(1)));
                }
            }
        }

        if (pts_3_vector.size() < 6) {
            LOGW("SimpleVIO",
                 "Not enough points for solve pnp pts_3_vector size [%zu]",
                 pts_3_vector.size());

            return false;
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0,
                     0.0,
                     0.0,
                     0.0,
                     1.0,
                     0.0,
                     0.0,
                     0.0,
                     1.0);
        if (!cv::solvePnP(pts_3_vector,
                          pts_2_vector,
                          K,
                          cv::Mat(),
                          rvec,
                          t,
                          true)) {
            LOGW("SimpleVIO", "solve pnp fail!");

            return false;
        }

        cv::Mat r;
        cv::Rodrigues(rvec, r);

        Mat33 R_pnp;
        cv::cv2eigen(r, R_pnp);
        R_pnp = R_pnp.transpose().eval();

        Vec3 T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);

        frame_it->second.R = R_pnp * ric_[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (VisualInitialAlign()) {
        return true;
    } else {
        LOGW("SimpleVIO", "misalign visual structure with IMU");

        return false;
    }
}

bool Estimator::VisualInitialAlign() {
    Timer t_g;

    // solve scale
    VecX x;
    bool result = VisualIMUAlignment(params_->window_size,
                                     all_image_frame_,
                                     Bgs_,
                                     G_,
                                     x,
                                     params_->G,
                                     tic_[0]);
    if (!result) {
        LOGW("SimpleVIO", "Solve G Failed!");

        return false;
    }

    // change state
    for (int i = 0; i <= frame_count_; ++i) {
        Rs_[i] = all_image_frame_[headers_[i]].R;
        Ps_[i] = all_image_frame_[headers_[i]].T;

        all_image_frame_[headers_[i]].is_key_frame = true;
    }

    VecX depth = feature_manager_.GetDepthVector();
    for (Eigen::Index i = 0; i < depth.size(); ++i) {
        depth[i] = -1.0;
    }
    feature_manager_.SetDepth(depth);

    // triangulat on cam pose , no tic
    std::vector<Vec3> temp_tic(params_->num_of_camera);
    for (int i = 0; i < params_->num_of_camera; ++i) {
        temp_tic[i].setZero();
    }

    // reset ric
    ric_[0] = params_->initial_ric[0];
    feature_manager_.SetRic(ric_);
    feature_manager_.Triangulate(Ps_, temp_tic, ric_);

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= params_->window_size; ++i) {
        // repropagate with only bg
        pre_integrations_[i]->Repropagate(Vec3::Zero(), Bgs_[i]);
    }

    for (int i = frame_count_; i >= 0; --i) {
        // set 0 as origin
        Ps_[i] = (s * Ps_[i] - Rs_[i] * params_->initial_tic[0]) -
                 (s * Ps_[0] - Rs_[0] * params_->initial_tic[0]);
    }

    // update velocity
    int kv = 0;
    std::map<double, ImageIMUFrame>::iterator frame_i;
    for (frame_i = all_image_frame_.begin(); frame_i != all_image_frame_.end();
         ++frame_i) {
        if (frame_i->second.is_key_frame) {
            Vs_[kv] = frame_i->second.R * x.segment<3>(kv * 3);
            kv += 1;
        }
    }

    // update scale
    feature_manager_.ScaleDepth(s);

    // update gravity
    Mat33 R0 = g2R(G_);
    R0       = ypr2R(Vec3{-R2ypr(R0 * Rs_[0]).x(), 0, 0}) * R0;
    G_       = R0 * G_;

    LOGD("SimpleVIO", "G (%lf %lf %lf)", G_(0), G_(1), G_(2));

    Mat33 rot_diff = R0;
    // Mat33 rot_diff = R0 * Rs[0].transpose();

    for (int i = 0; i <= frame_count_; ++i) {
        Ps_[i] = rot_diff * Ps_[i];
        Rs_[i] = rot_diff * Rs_[i];
        Vs_[i] = rot_diff * Vs_[i];
    }

    LOGD("SimpleVIO", "VisualInitialAlign cost %lf ms", t_g.End());

    return true;
}

bool Estimator::RelativePose(Mat33 &relative_R, Vec3 &relative_T, int &l) {
    // find previous frame which contians enough correspondance and parallex
    // with newest frame
    for (int i = 0; i < params_->window_size; ++i) {
        auto corres =
            feature_manager_.GetCorresponding(i, params_->window_size);
        if (corres.size() > 20) {
            double sum_parallax = 0.0;
            for (size_t j = 0; j < corres.size(); ++j) {
                Vec2 pts_0(corres[j].first(0), corres[j].first(1));
                Vec2 pts_1(corres[j].second(0), corres[j].second(1));

                sum_parallax += (pts_0 - pts_1).norm();
            }

            double average_parallax = sum_parallax / (double)(corres.size());
            if (average_parallax * 460 > 30 &&
                SolveRelativeRT(corres, relative_R, relative_T)) {
                l = i;

                return true;
            }
        }
    }

    return false;
}

void Estimator::SolveOdometry() {
    if (frame_count_ < params_->window_size) {
        return;
    }

    if (SolverFlag::NON_LINEAR == solver_flag_) {
        Timer t_tri;
        feature_manager_.Triangulate(Ps_, tic_, ric_);
        LOGD("SimpleVIO", "triangulation costs %lf ms", t_tri.End());

        BackendOptimization();
    }
}

void Estimator::Vector2Double() {
    for (int i = 0; i <= params_->window_size; ++i) {
        para_pose_[i][0] = Ps_[i].x();
        para_pose_[i][1] = Ps_[i].y();
        para_pose_[i][2] = Ps_[i].z();

        Qd q{Rs_[i]};
        para_pose_[i][3] = q.x();
        para_pose_[i][4] = q.y();
        para_pose_[i][5] = q.z();
        para_pose_[i][6] = q.w();

        para_speed_bias_[i][0] = Vs_[i].x();
        para_speed_bias_[i][1] = Vs_[i].y();
        para_speed_bias_[i][2] = Vs_[i].z();

        para_speed_bias_[i][3] = Bas_[i].x();
        para_speed_bias_[i][4] = Bas_[i].y();
        para_speed_bias_[i][5] = Bas_[i].z();

        para_speed_bias_[i][6] = Bgs_[i].x();
        para_speed_bias_[i][7] = Bgs_[i].y();
        para_speed_bias_[i][8] = Bgs_[i].z();
    }

    for (int i = 0; i < params_->num_of_camera; ++i) {
        para_extrinsic_[i][0] = tic_[i].x();
        para_extrinsic_[i][1] = tic_[i].y();
        para_extrinsic_[i][2] = tic_[i].z();

        Qd q{ric_[i]};
        para_extrinsic_[i][3] = q.x();
        para_extrinsic_[i][4] = q.y();
        para_extrinsic_[i][5] = q.z();
        para_extrinsic_[i][6] = q.w();
    }

    VecX depth = feature_manager_.GetDepthVector();
    for (int i = 0; i < feature_manager_.GetFeatureCount(); ++i) {
        para_feature_[i][0] = depth(i);
    }

    if (0 != params_->estimate_td) {
        para_td_[0][0] = td_;
    }
}

void Estimator::Double2Vector() {
    Vec3 origin_R0 = R2ypr(Rs_[0]);
    Vec3 origin_P0 = Ps_[0];

    if (failure_occur_) {
        origin_R0 = R2ypr(last_R0_);
        origin_P0 = last_P0_;

        failure_occur_ = false;
    }

    Vec3 origin_R00 = R2ypr(Qd(para_pose_[0][6],
                               para_pose_[0][3],
                               para_pose_[0][4],
                               para_pose_[0][5])
                                .toRotationMatrix());

    double y_diff = origin_R0.x() - origin_R00.x();

    // TODO
    Mat33 rot_diff = ypr2R(Vec3(y_diff, 0.0, 0.0));
    if (std::abs(std::abs(origin_R0.y()) - 90) < 1.0 ||
        std::abs(std::abs(origin_R00.y()) - 90) < 1.0) {
        LOGW("SimpleVIO", "euler singular point!");
        rot_diff = Rs_[0] * Qd(para_pose_[0][6],
                               para_pose_[0][3],
                               para_pose_[0][4],
                               para_pose_[0][5])
                                .toRotationMatrix()
                                .transpose();
    }

    for (int i = 0; i <= params_->window_size; ++i) {
        Rs_[i] = rot_diff * Qd(para_pose_[i][6],
                               para_pose_[i][3],
                               para_pose_[i][4],
                               para_pose_[i][5])
                                .normalized()
                                .toRotationMatrix();

        Ps_[i] = rot_diff * Vec3(para_pose_[i][0] - para_pose_[0][0],
                                 para_pose_[i][1] - para_pose_[0][1],
                                 para_pose_[i][2] - para_pose_[0][2]) +
                 origin_P0;

        Vs_[i] = rot_diff * Vec3(para_speed_bias_[i][0],
                                 para_speed_bias_[i][1],
                                 para_speed_bias_[i][2]);

        Bas_[i] = Vec3(para_speed_bias_[i][3],
                       para_speed_bias_[i][4],
                       para_speed_bias_[i][5]);

        Bgs_[i] = Vec3(para_speed_bias_[i][6],
                       para_speed_bias_[i][7],
                       para_speed_bias_[i][8]);
    }

    for (int i = 0; i < params_->num_of_camera; ++i) {
        tic_[i] = Vec3(para_extrinsic_[i][0],
                       para_extrinsic_[i][1],
                       para_extrinsic_[i][2]);

        ric_[i] = Qd(para_extrinsic_[i][6],
                     para_extrinsic_[i][3],
                     para_extrinsic_[i][4],
                     para_extrinsic_[i][5])
                      .normalized()
                      .toRotationMatrix();
    }

    VecX depth = feature_manager_.GetDepthVector();
    for (int i = 0; i < feature_manager_.GetFeatureCount(); ++i) {
        depth(i) = para_feature_[i][0];
    }
    feature_manager_.SetDepth(depth, true);

    if (0 != params_->estimate_td) {
        td_ = para_td_[0][0];
    }
}

bool Estimator::FailureDetection() {
    if (feature_manager_.GetLastTrackNum() < 2) {
        LOGW("SimpleVIO",
             "little feature %d",
             feature_manager_.GetLastTrackNum());

        // return true;
    }

    if (Bas_[params_->window_size].norm() > 2.5) {
        LOGW("SimpleVIO",
             "big IMU acc bias estimation %lf",
             Bas_[params_->window_size].norm());

        return true;
    }

    if (Bgs_[params_->window_size].norm() > 1.0) {
        LOGW("SimpleVIO",
             "big IMU gyr bias estimation %lf",
             Bgs_[params_->window_size].norm());

        return true;
    }

    Vec3 tmp_P = Ps_[params_->window_size];
    if ((tmp_P - last_P_).norm() > 5) {
        LOGW("SimpleVIO", "big translation");

        return true;
    }

    if (std::abs(tmp_P.z() - last_P_.z()) > 1) {
        LOGW("SimpleVIO", "big z translation");

        return true;
    }

    Mat33 tmp_R   = Rs_[params_->window_size];
    Mat33 delta_R = tmp_R.transpose() * last_R_;
    Qd delta_Q(delta_R);
    double delta_angle = std::acos(delta_Q.w()) * 2.0 * 180.0 / M_PI;
    if (delta_angle > 50) {
        LOGW("SimpleVIO", "big delta_angle");

        // return true;
    }

    return false;
}

void Estimator::MargOldFrame() {
    std::shared_ptr<LossFunction> loss_function =
        std::make_shared<CauchyLoss>(1.0);

    //  problem
    ProblemPtr problem =
        CreateProblemPtr((Problem::OptimizationMethod)params_->opt_method,
                         Problem::ProblemType::SLAM_PROBLEM);

    std::vector<std::shared_ptr<VertexPose>> vertices_pose;
    std::vector<std::shared_ptr<VertexSpeedBias>> vertices_speed_bias;

    int pose_dim = 0;

    // extrinsic as first certex
    std::shared_ptr<VertexPose> vertex_extrinsic =
        std::make_shared<VertexPose>();
    {
        vertex_extrinsic->SetParameters(para_extrinsic_[0].data());
        problem->AddVertex(vertex_extrinsic);
        pose_dim += vertex_extrinsic->LocalDimension();
    }

    // poses
    for (int i = 0; i < params_->window_size + 1; ++i) {
        // camera pose
        std::shared_ptr<VertexPose> vertex_pose =
            std::make_shared<VertexPose>();
        vertex_pose->SetParameters(para_pose_[i].data());
        vertices_pose.push_back(vertex_pose);
        problem->AddVertex(vertex_pose);
        pose_dim += vertex_pose->LocalDimension();

        // speed, acc bias, gyo bias
        std::shared_ptr<VertexSpeedBias> vertex_speed_bias =
            std::make_shared<VertexSpeedBias>();
        vertex_speed_bias->SetParameters(para_speed_bias_[i].data());
        vertices_speed_bias.push_back(vertex_speed_bias);
        problem->AddVertex(vertex_speed_bias);
        pose_dim += vertex_speed_bias->LocalDimension();
    }

    // IMU
    if (pre_integrations_[1]->GetSumDeltaTime() > 10.0) {
        LOGW("SimpleVIO", "preintegration dt is too long (> 10.0)");
    } else {
        std::shared_ptr<EdgeImu> edge_imu =
            std::make_shared<EdgeImu>(pre_integrations_[1]);
        edge_imu->SetVertex(0, vertices_pose[0]);
        edge_imu->SetVertex(1, vertices_speed_bias[0]);
        edge_imu->SetVertex(2, vertices_pose[1]);
        edge_imu->SetVertex(3, vertices_speed_bias[1]);
        problem->AddEdge(edge_imu);
    }

    // Visual Factor
    std::vector<std::shared_ptr<VertexInverseDepth>> vertices_inverse_depth;
    {
        int feature_index = 0;

        // TODO: update first
        auto &features = feature_manager_.GetFeatures();
        for (auto &f_per_id : features) {
            if (f_per_id->feature_per_frame_vec.size() >= 2 &&
                f_per_id->start_frame < params_->window_size - 2) {
                int imu_i = f_per_id->start_frame;
                int imu_j = imu_i - 1;

                std::shared_ptr<VertexInverseDepth> vertex_inverse_depth =
                    std::make_shared<VertexInverseDepth>();
                vertex_inverse_depth->SetParameters(
                    para_feature_[feature_index].data());
                problem->AddVertex(vertex_inverse_depth);
                vertices_inverse_depth.push_back(vertex_inverse_depth);

                feature_index += 1;

                Vec3 pts_i = f_per_id->feature_per_frame_vec[0]
                                 .feature_observation.point;

                for (auto &f_per_frame : f_per_id->feature_per_frame_vec) {
                    imu_j += 1;
                    if (imu_i == imu_j) {
                        continue;
                    }

                    Vec3 pts_j = f_per_frame.feature_observation.point;

                    std::shared_ptr<EdgeReprojection> edge_reprojection =
                        std::make_shared<EdgeReprojection>(pts_i, pts_j);
                    edge_reprojection->SetVertex(0, vertex_inverse_depth);
                    edge_reprojection->SetVertex(1, vertices_pose[imu_i]);
                    edge_reprojection->SetVertex(2, vertices_pose[imu_j]);
                    edge_reprojection->SetVertex(3, vertex_extrinsic);

                    Mat22 info =
                        project_sqrt_info_.transpose() * project_sqrt_info_;
                    edge_reprojection->SetInformation(info.data());
                    edge_reprojection->SetLossFunction(loss_function);
                    problem->AddEdge(edge_reprojection);
                }
            }
        }
    }

    // prior
    if (Hprior_.rows() > 0) {
        problem->SetHessianPrior(Hprior_);
        problem->SetErrPrior(errprior_);
        problem->SetJtPrior(Jprior_inv_);
        problem->ExtendHessiansPriorSize(15);
    } else {
        Hprior_ = MatXX::Zero(pose_dim, pose_dim);
        bprior_ = VecX::Zero(pose_dim);
        problem->SetHessianPrior(Hprior_);
        problem->SetbPrior(bprior_);
    }

    std::vector<std::shared_ptr<BaseVertex>> marg_vertices;
    marg_vertices.push_back(vertices_pose[0]);
    marg_vertices.push_back(vertices_speed_bias[0]);

    problem->Marginalize(marg_vertices, 0, pose_dim);

    // update prior
    Hprior_     = problem->GetHessianPrior();
    bprior_     = problem->GetbPrior();
    errprior_   = problem->GetErrPrior();
    Jprior_inv_ = problem->GetJtPrior();
}

void Estimator::MargNewFrame() {
    //  problem
    ProblemPtr problem =
        CreateProblemPtr((Problem::OptimizationMethod)params_->opt_method,
                         Problem::ProblemType::SLAM_PROBLEM);

    std::vector<std::shared_ptr<VertexPose>> vertices_pose;
    std::vector<std::shared_ptr<VertexSpeedBias>> vertices_speed_bias;

    int pose_dim = 0;

    // extrinsic as first certex
    std::shared_ptr<VertexPose> vertex_extrinsic =
        std::make_shared<VertexPose>();
    {
        vertex_extrinsic->SetParameters(para_extrinsic_[0].data());
        problem->AddVertex(vertex_extrinsic);
        pose_dim += vertex_extrinsic->LocalDimension();
    }

    // poses
    for (int i = 0; i < params_->window_size + 1; ++i) {
        // camera pose
        std::shared_ptr<VertexPose> vertex_pose =
            std::make_shared<VertexPose>();
        vertex_pose->SetParameters(para_pose_[i].data());
        vertices_pose.push_back(vertex_pose);
        problem->AddVertex(vertex_pose);
        pose_dim += vertex_pose->LocalDimension();

        // speed, acc bias, gyo bias
        std::shared_ptr<VertexSpeedBias> vertex_speed_bias =
            std::make_shared<VertexSpeedBias>();
        vertex_speed_bias->SetParameters(para_speed_bias_[i].data());
        vertices_speed_bias.push_back(vertex_speed_bias);
        problem->AddVertex(vertex_speed_bias);
        pose_dim += vertex_speed_bias->LocalDimension();
    }

    // prior
    if (Hprior_.rows() > 0) {
        problem->SetHessianPrior(Hprior_);
        problem->SetbPrior(bprior_);
        problem->SetErrPrior(errprior_);
        problem->SetJtPrior(Jprior_inv_);
        problem->ExtendHessiansPriorSize(15);
    } else {
        Hprior_ = MatXX::Zero(pose_dim, pose_dim);
        bprior_ = VecX::Zero(pose_dim);
        problem->SetHessianPrior(Hprior_);
        problem->SetbPrior(bprior_);
    }

    std::vector<std::shared_ptr<BaseVertex>> marg_vertices;
    marg_vertices.push_back(vertices_pose[params_->window_size - 1]);
    marg_vertices.push_back(vertices_speed_bias[params_->window_size - 1]);

    problem->Marginalize(marg_vertices, 0, pose_dim);

    // update prior
    Hprior_     = problem->GetHessianPrior();
    bprior_     = problem->GetbPrior();
    errprior_   = problem->GetErrPrior();
    Jprior_inv_ = problem->GetJtPrior();
}

void Estimator::ProblemSolve() {
    std::shared_ptr<LossFunction> loss_function =
        std::make_shared<CauchyLoss>(1.0);

    //  problem
    ProblemPtr problem =
        CreateProblemPtr((Problem::OptimizationMethod)params_->opt_method,
                         Problem::ProblemType::SLAM_PROBLEM);

    std::vector<std::shared_ptr<VertexPose>> vertices_pose;
    std::vector<std::shared_ptr<VertexSpeedBias>> vertices_speed_bias;

    int pose_dim = 0;

    // extrinsic as first certex
    std::shared_ptr<VertexPose> vertex_extrinsic =
        std::make_shared<VertexPose>();
    {
        if (0 == estimate_extrinsic_) {
            LOGD("SimpleVIO", "fix extinsic param");

            //  TODO:: set Hessian prior to zero
            vertex_extrinsic->SetFixed();
        } else {
            LOGD("SimpleVIO", "estimate extinsic param");
        }

        vertex_extrinsic->SetParameters(para_extrinsic_[0].data());
        problem->AddVertex(vertex_extrinsic);
        pose_dim += vertex_extrinsic->LocalDimension();
    }

    // poses
    for (int i = 0; i < params_->window_size + 1; ++i) {
        // camera pose
        std::shared_ptr<VertexPose> vertex_pose =
            std::make_shared<VertexPose>();
        vertex_pose->SetParameters(para_pose_[i].data());
        vertices_pose.push_back(vertex_pose);
        problem->AddVertex(vertex_pose);
        pose_dim += vertex_pose->LocalDimension();

        // speed, acc bias, gyo bias
        std::shared_ptr<VertexSpeedBias> vertex_speed_bias =
            std::make_shared<VertexSpeedBias>();
        vertex_speed_bias->SetParameters(para_speed_bias_[i].data());
        vertices_speed_bias.push_back(vertex_speed_bias);
        problem->AddVertex(vertex_speed_bias);
        pose_dim += vertex_speed_bias->LocalDimension();
    }

    // IMU
    for (int i = 0; i < params_->window_size; ++i) {
        int j = i + 1;
        if (pre_integrations_[j]->GetSumDeltaTime() > 10.0) {
            LOGW("SimpleVIO", "preintegration dt is too long (> 10.0)");
            continue;
        }

        std::shared_ptr<EdgeImu> edge_imu =
            std::make_shared<EdgeImu>(pre_integrations_[j]);
        edge_imu->SetVertex(0, vertices_pose[i]);
        edge_imu->SetVertex(1, vertices_speed_bias[i]);
        edge_imu->SetVertex(2, vertices_pose[j]);
        edge_imu->SetVertex(3, vertices_speed_bias[j]);
        problem->AddEdge(edge_imu);
    }

    // Visual Factor
    std::vector<std::shared_ptr<VertexInverseDepth>> vertices_inverse_depth;
    {
        int feature_index = 0;

        auto &features = feature_manager_.GetFeatures();
        for (auto &f_per_id : features) {
            if (f_per_id->feature_per_frame_vec.size() >= 2 &&
                f_per_id->start_frame < params_->window_size - 2) {
                int imu_i = f_per_id->start_frame;
                int imu_j = imu_i;

                std::shared_ptr<VertexInverseDepth> vertex_inverse_depth =
                    std::make_shared<VertexInverseDepth>();
                vertex_inverse_depth->SetParameters(
                    para_feature_[feature_index].data());
                problem->AddVertex(vertex_inverse_depth);
                vertices_inverse_depth.push_back(vertex_inverse_depth);

                feature_index += 1;

                Vec3 pts_i = f_per_id->feature_per_frame_vec[0]
                                 .feature_observation.point;

                for (auto &f_per_frame : f_per_id->feature_per_frame_vec) {
                    if (imu_i != imu_j) {
                        Vec3 pts_j = f_per_frame.feature_observation.point;

                        std::shared_ptr<EdgeReprojection> edge_reprojection =
                            std::make_shared<EdgeReprojection>(pts_i, pts_j);
                        edge_reprojection->SetVertex(0, vertex_inverse_depth);
                        edge_reprojection->SetVertex(1, vertices_pose[imu_i]);
                        edge_reprojection->SetVertex(2, vertices_pose[imu_j]);
                        edge_reprojection->SetVertex(3, vertex_extrinsic);

                        Mat22 info =
                            project_sqrt_info_.transpose() * project_sqrt_info_;
                        edge_reprojection->SetInformation(info.data());
                        edge_reprojection->SetLossFunction(loss_function);
                        problem->AddEdge(edge_reprojection);
                    }

                    imu_j += 1;
                }
            }
        }
    }

    // prior
    if (Hprior_.rows() > 0) {
        problem->SetHessianPrior(Hprior_);
        problem->SetbPrior(bprior_);
        problem->SetErrPrior(errprior_);
        problem->SetJtPrior(Jprior_inv_);
        problem->ExtendHessiansPriorSize(15);
    }

    problem->Solve(30);

    // update bprior_, Hprior_ do not need update
    if (Hprior_.rows() > 0) {
        LOGD("SimpleVIO", "----------- update bprior -------------");
        LOGD("SimpleVIO",
             "    before: %lf, %lf",
             bprior_.norm(),
             errprior_.norm());

        bprior_   = problem->GetbPrior();
        errprior_ = problem->GetErrPrior();

        LOGD("SimpleVIO",
             "    after: %lf, %lf",
             bprior_.norm(),
             errprior_.norm());
    }

    // update parameter
    for (int i = 0; i < params_->window_size + 1; ++i) {
        vertices_pose[i]->GetParameters(para_pose_[i].data());
        vertices_speed_bias[i]->GetParameters(para_speed_bias_[i].data());
    }

    for (size_t i = 0; i < vertices_inverse_depth.size(); ++i) {
        vertices_inverse_depth[i]->GetParameters(para_feature_[i].data());
    }
}

void Estimator::BackendOptimization() {
    Timer t_solver;

    Vector2Double();

    ProblemSolve();

    Double2Vector();

    LOGD("SimpleVIO", "whole time for solver %lf ms", t_solver.End());

    Timer t_whole_marginalization;
    if (MarginalizationFlag::MARGIN_OLD == marginalization_flag_) {
        Vector2Double();

        MargOldFrame();

    } else {
        if (Hprior_.rows() > 0) {
            Vector2Double();

            MargNewFrame();
        }
    }

    LOGD("SimpleVIO",
         "whole time for marginalize %lf ms",
         t_whole_marginalization.End());
}

void Estimator::SlideWindow() {
    Timer t_margin;

    if (MarginalizationFlag::MARGIN_OLD == marginalization_flag_) {
        double t_0 = headers_[0];
        back_R0_   = Rs_[0];
        back_P0_   = Ps_[0];
        if (params_->window_size == frame_count_) {
            for (int i = 0; i < params_->window_size; ++i) {
                Rs_[i].swap(Rs_[i + 1]);

                std::swap(pre_integrations_[i], pre_integrations_[i + 1]);

                dt_buf_[i].swap(dt_buf_[i + 1]);
                linear_acceleration_buf_[i].swap(
                    linear_acceleration_buf_[i + 1]);
                angular_velocity_buf_[i].swap(angular_velocity_buf_[i + 1]);

                headers_[i] = headers_[i + 1];

                Ps_[i].swap(Ps_[i + 1]);
                Vs_[i].swap(Vs_[i + 1]);
                Bas_[i].swap(Bas_[i + 1]);
                Bgs_[i].swap(Bgs_[i + 1]);
            }

            headers_[params_->window_size] = headers_[params_->window_size - 1];
            Ps_[params_->window_size]      = Ps_[params_->window_size - 1];
            Vs_[params_->window_size]      = Vs_[params_->window_size - 1];
            Rs_[params_->window_size]      = Rs_[params_->window_size - 1];
            Bas_[params_->window_size]     = Bas_[params_->window_size - 1];
            Bgs_[params_->window_size]     = Bgs_[params_->window_size - 1];

            pre_integrations_[params_->window_size].reset(
                new IMUIntegration(acc_0_,
                                   gyr_0_,
                                   Bas_[params_->window_size],
                                   Bgs_[params_->window_size],
                                   params_->G,
                                   params_->acc_sigma_n,
                                   params_->gyr_sigma_n,
                                   params_->acc_sigma_w,
                                   params_->gyr_sigma_w));

            dt_buf_[params_->window_size].clear();
            linear_acceleration_buf_[params_->window_size].clear();
            angular_velocity_buf_[params_->window_size].clear();

            {
                // clean from start to t0
                auto it_0 = all_image_frame_.find(t_0);

                it_0->second.pre_integration.reset();

                for (auto iter = all_image_frame_.begin(); iter != it_0;
                     ++iter) {
                    iter->second.pre_integration.reset();
                }

                all_image_frame_.erase(all_image_frame_.begin(), it_0);
                all_image_frame_.erase(t_0);
            }

            SlideWindowOld();
        }
    } else {
        if (params_->window_size == frame_count_) {
            for (size_t i = 0; i < dt_buf_[frame_count_].size(); ++i) {
                double tmp_dt = dt_buf_[frame_count_][i];
                Vec3 tmp_linear_acceleration =
                    linear_acceleration_buf_[frame_count_][i];
                Vec3 tmp_angular_velocity =
                    angular_velocity_buf_[frame_count_][i];

                pre_integrations_[frame_count_ - 1]->AddMeasure(
                    tmp_dt,
                    tmp_linear_acceleration,
                    tmp_angular_velocity);

                dt_buf_[frame_count_ - 1].push_back(tmp_dt);
                linear_acceleration_buf_[frame_count_ - 1].push_back(
                    tmp_linear_acceleration);
                angular_velocity_buf_[frame_count_ - 1].push_back(
                    tmp_angular_velocity);
            }

            headers_[frame_count_ - 1] = headers_[frame_count_];
            Ps_[frame_count_ - 1]      = Ps_[frame_count_];
            Vs_[frame_count_ - 1]      = Vs_[frame_count_];
            Rs_[frame_count_ - 1]      = Rs_[frame_count_];
            Bas_[frame_count_ - 1]     = Bas_[frame_count_];
            Bgs_[frame_count_ - 1]     = Bgs_[frame_count_];

            pre_integrations_[params_->window_size].reset(
                new IMUIntegration(acc_0_,
                                   gyr_0_,
                                   Bas_[params_->window_size],
                                   Bgs_[params_->window_size],
                                   params_->G,
                                   params_->acc_sigma_n,
                                   params_->gyr_sigma_n,
                                   params_->acc_sigma_w,
                                   params_->gyr_sigma_w));

            dt_buf_[params_->window_size].clear();
            linear_acceleration_buf_[params_->window_size].clear();
            angular_velocity_buf_[params_->window_size].clear();

            SlideWindowNew();
        }
    }

    LOGD("SimpleVIO", "SlideWindow cost %lf ms", t_margin.End());
}

// real marginalization is removed in solve_ceres()
void Estimator::SlideWindowNew() {
    feature_manager_.RemoveFront(frame_count_);
}

// real marginalization is removed in solve_ceres()
void Estimator::SlideWindowOld() {
    bool shift_depth = (SolverFlag::NON_LINEAR == solver_flag_);
    if (shift_depth) {
        Mat33 R0 = back_R0_ * ric_[0];
        Mat33 R1 = Rs_[0] * ric_[0];
        Vec3 P0  = back_P0_ + back_R0_ * tic_[0];
        Vec3 P1  = Ps_[0] + Rs_[0] * tic_[0];

        feature_manager_.RemoveBackShiftDepth(R0, P0, R1, P1);
    } else {
        feature_manager_.RemoveBack();
    }
}

}  // namespace SimpleVIO