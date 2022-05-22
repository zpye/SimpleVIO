#include "global_sfm.h"

#include <cassert>
#include <iostream>
#include <memory>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "backend/edge_reprojection.h"
#include "backend/vertex_point_xyz.h"
#include "backend/vertex_pose.h"
#include "utils/logger.h"

// #define USE_CERES
#if defined(USE_CERES)
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif

namespace SimpleVIO {

#if defined(USE_CERES)
struct ReprojectionError3D {
    ReprojectionError3D(double observed_u, double observed_v)
        : observed_u(observed_u), observed_v(observed_v) {}

    template<typename T>
    bool operator()(const T *const camera_R,
                    const T *const camera_T,
                    const T *point,
                    T *residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp         = p[0] / p[2];
        T yp         = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y) {
        return (
            new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
                new ReprojectionError3D(observed_x, observed_y)));
    }

    double observed_u;
    double observed_v;
};
#endif

bool GlobalSFM::ConstructSFM(const int frame_num,
                             const int start_frame_idx,
                             const Mat33 &relative_R,
                             const Vec3 &relative_T,
                             std::vector<Qd> &q,
                             std::vector<Vec3> &T,
                             std::vector<SFMFeature> &sfm_f,
                             std::map<int, Vec3> &sfm_tracked_points) {
    const int l = start_frame_idx;

    //  intial two view with relative_R relative_T
    q[l].setIdentity();
    T[l].setZero();
    q[frame_num - 1] = q[l] * Qd(relative_R);
    T[frame_num - 1] = relative_T;

    // rotate to cam frame
    std::vector<Mat33> c_Rotation(frame_num);
    std::vector<Vec3> c_Translation(frame_num);
    std::vector<Qd> c_Quat(frame_num);
    std::vector<Mat34> poses(frame_num);

    c_Quat[l]                  = q[l].inverse();
    c_Rotation[l]              = c_Quat[l].toRotationMatrix();
    c_Translation[l]           = -c_Rotation[l] * T[l];
    poses[l].block<3, 3>(0, 0) = c_Rotation[l];
    poses[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quat[frame_num - 1]     = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] =
        -c_Rotation[frame_num - 1] * T[frame_num - 1];
    poses[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    poses[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 1: trangulate between l ----- frame_num - 1
    // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
    for (int i = l; i < frame_num - 1; ++i) {
        // solve pnp
        if (i > l) {
            Mat33 initial_R = c_Rotation[i - 1];
            Vec3 initial_P  = c_Translation[i - 1];
            if (!SolveFrameByPnP(i, initial_R, initial_P, sfm_f)) {
                return false;
            }

            c_Rotation[i]              = initial_R;
            c_Translation[i]           = initial_P;
            c_Quat[i]                  = c_Rotation[i];
            poses[i].block<3, 3>(0, 0) = c_Rotation[i];
            poses[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        // triangulate point based on the solve pnp result
        TriangulateTwoFrames(i,
                             poses[i],
                             frame_num - 1,
                             poses[frame_num - 1],
                             sfm_f);
    }

    // 3: triangulate l-----l+1 l+2 ... frame_num -2
    for (int i = l + 1; i < frame_num - 1; ++i) {
        TriangulateTwoFrames(l, poses[l], i, poses[i], sfm_f);
    }

    // 4: solve pnp l-1; triangulate l-1 ----- l
    //              l-2              l-2 ----- l
    for (int i = l - 1; i >= 0; --i) {
        // solve pnp
        Mat33 initial_R = c_Rotation[i + 1];
        Vec3 initial_P  = c_Translation[i + 1];
        if (!SolveFrameByPnP(i, initial_R, initial_P, sfm_f)) {
            return false;
        }

        c_Rotation[i]              = initial_R;
        c_Translation[i]           = initial_P;
        c_Quat[i]                  = c_Rotation[i];
        poses[i].block<3, 3>(0, 0) = c_Rotation[i];
        poses[i].block<3, 1>(0, 3) = c_Translation[i];

        // triangulate
        TriangulateTwoFrames(i, poses[i], l, poses[l], sfm_f);
    }

    // 5: triangulate all other points
    for (size_t i = 0; i < sfm_f.size(); ++i) {
        if (sfm_f[i].state) {
            continue;
        }

        if (sfm_f[i].observations.size() >= 2) {
            Vec2 point0;
            Vec2 point1;
            int frame_0 = sfm_f[i].observations[0].first;
            point0      = sfm_f[i].observations[0].second;
            int frame_1 = sfm_f[i].observations.back().first;
            point1      = sfm_f[i].observations.back().second;

            Vec3 point_3d;
            TriangulateOnePoint(poses[frame_0],
                                poses[frame_1],
                                point0,
                                point1,
                                point_3d);

            sfm_f[i].state = true;

            sfm_f[i].position[0] = point_3d(0);
            sfm_f[i].position[1] = point_3d(1);
            sfm_f[i].position[2] = point_3d(2);

            std::cout << "trangulated : " << frame_0 << " " << frame_1
                      << "  3d point : " << i << "  " << point_3d.transpose()
                      << std::endl;
        }
    }

    // for (int i = 0; i < frame_num; i++) {
    //     q[i] = c_Rotation[i].transpose();
    //     std::cout << "solvePnP  q"
    //               << " i " << i << "  " << q[i].w() << "  "
    //               << q[i].vec().transpose() << std::endl;
    // }
    // for (int i = 0; i < frame_num; i++) {
    //     Vec3 t_tmp = -1 * (q[i] * c_Translation[i]);
    //     std::cout << "solvePnP  t"
    //               << " i " << i << "  " << t_tmp.x() << "  " << t_tmp.y()
    //               << "  " << t_tmp.z() << std::endl;
    // }

    std::vector<std::vector<double>> c_rotation(frame_num,
                                                std::vector<double>(4));
    std::vector<std::vector<double>> c_translation(frame_num,
                                                   std::vector<double>(3));

    std::vector<std::vector<double>> optimize_poses(frame_num,
                                                    std::vector<double>(7));

    // full BA
#if defined(USE_CERES)
    ceres::Problem problem;

    ceres::LocalParameterization *local_parameterization =
        new ceres::QuaternionParameterization();

    for (int i = 0; i < frame_num; ++i) {
        // double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0]    = c_Quat[i].w();
        c_rotation[i][1]    = c_Quat[i].x();
        c_rotation[i][2]    = c_Quat[i].y();
        c_rotation[i][3]    = c_Quat[i].z();

        problem.AddParameterBlock(c_rotation[i].data(),
                                  4,
                                  local_parameterization);
        problem.AddParameterBlock(c_translation[i].data(), 3);

        if (i == l) {
            problem.SetParameterBlockConstant(c_rotation[i].data());
        }

        if (i == l || i == frame_num - 1) {
            problem.SetParameterBlockConstant(c_translation[i].data());
        }
    }

    for (size_t i = 0; i < sfm_f.size(); ++i) {
        if (!sfm_f[i].state) {
            continue;
        }

        for (size_t k = 0; k < sfm_f[i].observations.size(); ++k) {
            const int idx = sfm_f[i].observations[k].first;

            ceres::CostFunction *cost_function = ReprojectionError3D::Create(
                sfm_f[i].observations[k].second.x(),
                sfm_f[i].observations[k].second.y());

            problem.AddResidualBlock(cost_function,
                                     nullptr,
                                     c_rotation[idx].data(),
                                     c_translation[idx].data(),
                                     sfm_f[i].position);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.2;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOGD("SimpleVIO", "BriefReport:\n%s\n", summary.BriefReport().c_str());

    if (ceres::CONVERGENCE == summary.termination_type ||
        summary.final_cost < 5e-03) {
        LOGD("SimpleVIO", "vision only BA converge");
    } else {
        LOGD("SimpleVIO", "vision only BA not converge");

        return false;
    }

    for (int i = 0; i < frame_num; ++i) {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i]     = q[i].inverse().normalized();

        T[i] = -(q[i] * Vec3(c_translation[i][0],
                             c_translation[i][1],
                             c_translation[i][2]));
    }

    for (auto &f : sfm_f) {
        if (f.state) {
            sfm_tracked_points[f.id] =
                Vec3(f.position[0], f.position[1], f.position[2]);
        }
    }
#else
    // TODO: other optimization methods
    ProblemPtr problem =
        CreateProblemPtr(Problem::OptimizationMethod::LM_NIELSON,
                         Problem::ProblemType::SLAM_PROBLEM);

    std::vector<std::shared_ptr<VertexPose>> vertices_pose;

    // add vertices
    for (int i = 0; i < frame_num; ++i) {
        // double array for optimizer
        //  tx, ty, tz, qx, qy, qz, qw
        optimize_poses[i][0] = c_Translation[i].x();
        optimize_poses[i][1] = c_Translation[i].y();
        optimize_poses[i][2] = c_Translation[i].z();
        optimize_poses[i][3] = c_Quat[i].x();
        optimize_poses[i][4] = c_Quat[i].y();
        optimize_poses[i][5] = c_Quat[i].z();
        optimize_poses[i][6] = c_Quat[i].w();

        std::shared_ptr<VertexPose> vertex_pose =
            std::make_shared<VertexPose>();
        vertex_pose->SetParameters(optimize_poses[i].data());
        LOGD("SimpleVIO",
             "pose0 (%lf %lf %lf %lf %lf %lf %lf)",
             optimize_poses[i][0],
             optimize_poses[i][1],
             optimize_poses[i][2],
             optimize_poses[i][3],
             optimize_poses[i][4],
             optimize_poses[i][5],
             optimize_poses[i][6]);

        if (l == i) {
            // fix rotation and translation
            vertex_pose->SetFixed(true);
        }

        if (frame_num - 1 == i) {
            // fix translation only
            vertex_pose->SetFixTranslation(true);
        }

        vertices_pose.push_back(vertex_pose);

        problem->AddVertex(vertex_pose);
    }

    std::vector<std::shared_ptr<VertexPointXYZ>> vertices_point_xyz;
    std::vector<std::shared_ptr<EdgeReprojectionXYZ>> edges_reprojection_xyz;

    // add edges
    for (size_t i = 0; i < sfm_f.size(); ++i) {
        if (!sfm_f[i].state) {
            continue;
        }

        // vertex point xyz
        std::shared_ptr<VertexPointXYZ> vertex_point_xyz =
            std::make_shared<VertexPointXYZ>();
        vertex_point_xyz->SetParameters(sfm_f[i].position);

        vertices_point_xyz.push_back(vertex_point_xyz);

        problem->AddVertex(vertex_point_xyz);

        // edges
        for (size_t k = 0; k < sfm_f[i].observations.size(); ++k) {
            int pose_idx = sfm_f[i].observations[k].first;

            std::shared_ptr<EdgeReprojectionXYZ> edge_reprojection_xyz =
                std::make_shared<EdgeReprojectionXYZ>(
                    sfm_f[i].observations[k].second,
                    Vec3::Zero(),
                    Qd::Identity());

            edge_reprojection_xyz->SetVertex(0, vertex_point_xyz);
            edge_reprojection_xyz->SetVertex(1, vertices_pose[pose_idx]);

            edges_reprojection_xyz.push_back(edge_reprojection_xyz);

            problem->AddEdge(edge_reprojection_xyz);
        }
    }

    // TODO: convergence threshold, max solve time
    if (problem->Solve(30)) {
        LOGD("SimpleVIO", "vision only BA converge");
    } else {
        LOGD("SimpleVIO", "vision only BA not converge");

        return false;
    }

    // update poses
    for (int i = 0; i < frame_num; ++i) {
        vertices_pose[i]->GetParameters(optimize_poses[i].data());

        q[i].x() = optimize_poses[i][3];
        q[i].y() = optimize_poses[i][4];
        q[i].z() = optimize_poses[i][5];
        q[i].w() = optimize_poses[i][6];
        q[i]     = q[i].inverse();

        T[i] = -(q[i] * Vec3(optimize_poses[i][0],
                             optimize_poses[i][1],
                             optimize_poses[i][2]));

        LOGD("SimpleVIO",
             "%d SFM R(%lf %lf %lf %lf), T(%lf %lf %lf)",
             i,
             q[i].x(),
             q[i].y(),
             q[i].z(),
             q[i].w(),
             T[i].x(),
             T[i].y(),
             T[i].z());
    }

    int i = 0;
    for (auto &f : sfm_f) {
        if (f.state) {
            vertices_point_xyz[i]->GetParameters(f.position);

            sfm_tracked_points[f.id] =
                Vec3(f.position[0], f.position[1], f.position[2]);

            i += 1;
        }
    }
#endif

    return true;
}

bool GlobalSFM::SolveFrameByPnP(const int frame_idx,
                                Mat33 &initial_R,
                                Vec3 &initial_P,
                                std::vector<SFMFeature> &sfm_f) {
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (size_t i = 0; i < sfm_f.size(); ++i) {
        if (!sfm_f[i].state) {
            continue;
        }

        for (size_t k = 0; k < sfm_f[i].observations.size(); ++k) {
            if (frame_idx == sfm_f[i].observations[k].first) {
                Vec2 &img_pts = sfm_f[i].observations[k].second;

                pts_2_vector.push_back(cv::Point2f(img_pts(0), img_pts(1)));

                pts_3_vector.push_back(cv::Point3f(sfm_f[i].position[0],
                                                   sfm_f[i].position[1],
                                                   sfm_f[i].position[2]));

                break;
            }
        }
    }

    if (pts_2_vector.size() < 15) {
        LOGW("SimpleVIO",
             "unstable features tracking, please slowly move your device!");

        if (pts_2_vector.size() < 10) {
            LOGE("SimpleVIO",
                 "too few (< 10) features are tracking!!! [%zu]",
                 pts_2_vector.size());

            return false;
        }
    }

    cv::Mat temp_R;
    cv::eigen2cv(initial_R, temp_R);

    cv::Mat rvec;
    cv::Rodrigues(temp_R, rvec);

    cv::Mat t;
    cv::eigen2cv(initial_P, t);

    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    bool pnp_success =
        cv::solvePnP(pts_3_vector, pts_2_vector, K, cv::Mat(), rvec, t, true);
    if (!pnp_success) {
        LOGE("SimpleVIO", "solvePnP failed!!!");

        return false;
    }

    cv::Mat r;
    cv::Rodrigues(rvec, r);

    // update R and P
    cv::cv2eigen(r, initial_R);
    cv::cv2eigen(t, initial_P);

    return true;
}

void GlobalSFM::TriangulateOnePoint(const Mat34 &pose0,
                                    const Mat34 &pose1,
                                    const Vec2 &point0,
                                    const Vec2 &point1,
                                    Vec3 &point_3d) {
    Mat44 A  = Mat44::Zero();
    A.row(0) = point0(0) * pose0.row(2) - pose0.row(0);
    A.row(1) = point0(1) * pose0.row(2) - pose0.row(1);
    A.row(2) = point1(0) * pose1.row(2) - pose1.row(0);
    A.row(3) = point1(1) * pose1.row(2) - pose1.row(1);

    Vec4 point_homo = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d = point_homo.head<3>() / point_homo(3);
}

void GlobalSFM::TriangulateTwoFrames(const int frame0,
                                     const Mat34 &pose0,
                                     const int frame1,
                                     const Mat34 &pose1,
                                     std::vector<SFMFeature> &sfm_f) {
    assert(frame0 != frame1);
    for (size_t i = 0; i < sfm_f.size(); ++i) {
        if (sfm_f[i].state) {
            continue;
        }

        bool has_0 = false;
        bool has_1 = false;
        Vec2 point0;
        Vec2 point1;
        for (size_t k = 0; k < sfm_f[i].observations.size(); ++k) {
            if (frame0 == sfm_f[i].observations[k].first) {
                point0 = sfm_f[i].observations[k].second;
                has_0  = true;
            }

            if (frame1 == sfm_f[i].observations[k].first) {
                point1 = sfm_f[i].observations[k].second;
                has_1  = true;
            }
        }

        if (has_0 && has_1) {
            Vec3 point_3d;
            TriangulateOnePoint(pose0, pose1, point0, point1, point_3d);

            sfm_f[i].state       = true;
            sfm_f[i].position[0] = point_3d(0);
            sfm_f[i].position[1] = point_3d(1);
            sfm_f[i].position[2] = point_3d(2);
        }
    }
}

}  // namespace SimpleVIO