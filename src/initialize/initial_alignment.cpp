#include "initial_alignment.h"

#include <cmath>
#include <iterator>

#include "imu/imu_utils.h"
#include "utils/logger.h"

namespace SimpleVIO {

static void SolveGyroscopeBias(const size_t window_size,
                               std::map<double, ImageIMUFrame> &all_image_frame,
                               std::vector<Vec3> &Bgs) {
    Mat33 A;
    A.setZero();

    Vec3 b;
    b.setZero();

    std::map<double, ImageIMUFrame>::iterator frame_i;
    std::map<double, ImageIMUFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin();
         std::next(frame_i) != all_image_frame.end();
         ++frame_i) {
        frame_j = std::next(frame_i);

        Qd q_ij(frame_i->second.R.transpose() * frame_j->second.R);

        Mat33 tmp_A =
            frame_j->second.pre_integration->GetJacobian().block<3, 3>(
                StateOrder::O_R,
                StateOrder::O_BG);

        Vec3 tmp_b =
            2.0 *
            (frame_j->second.pre_integration->GetDeltaQ().inverse() * q_ij)
                .vec();

        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }

    Vec3 delta_bg = A.ldlt().solve(b);

    // update gyro bias
    for (size_t i = 0; i <= window_size; ++i) {
        Bgs[i] += delta_bg;
    }

    // repropagate
    for (frame_i = all_image_frame.begin();
         std::next(frame_i) != all_image_frame.end();
         ++frame_i) {
        frame_j = std::next(frame_i);
        frame_j->second.pre_integration->Repropagate(Vec3::Zero(), Bgs[0]);
    }
}

static Mat32 TangentBasis(const Vec3 &g0) {
    Vec3 a = g0.normalized();

    Vec3 tmp(0.0, 0.0, 1.0);
    if (a == tmp) {
        tmp << 1.0, 0.0, 0.0;
    }

    Vec3 b = (tmp - a * (a.transpose() * tmp)).normalized();
    Vec3 c = a.cross(b);

    Mat32 bc;
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;

    return bc;
}

void RefineGravity(std::map<double, ImageIMUFrame> &all_image_frame,
                   Vec3 &g,
                   VecX &x,
                   const Vec3 &G_const,
                   const Vec3 &tic0) {
    Vec3 g0 = g.normalized() * G_const.norm();

    size_t all_frame_count = all_image_frame.size();
    size_t n_state         = all_frame_count * 3 + 2 + 1;

    MatXX A(n_state, n_state);
    A.setZero();

    VecX b(n_state);
    b.setZero();

    std::map<double, ImageIMUFrame>::iterator frame_i;
    std::map<double, ImageIMUFrame>::iterator frame_j;
    for (int k = 0; k < 4; ++k) {
        Mat32 lxly = TangentBasis(g0);

        int i = 0;
        for (frame_i = all_image_frame.begin();
             std::next(frame_i) != all_image_frame.end();
             ++frame_i, ++i) {
            frame_j = next(frame_i);

            MatXX tmp_A(6, 9);
            tmp_A.setZero();

            VecX tmp_b(6);
            tmp_b.setZero();

            const double dt =
                frame_j->second.pre_integration->GetSumDeltaTime();

            tmp_A.block<3, 3>(0, 0) = -dt * Mat33::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt /
                                      2.0 * Mat33::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() *
                                      (frame_j->second.T - frame_i->second.T) /
                                      100.0;
            tmp_b.block<3, 1>(0, 0) =
                frame_j->second.pre_integration->GetDeltaP() +
                frame_i->second.R.transpose() * frame_j->second.R * tic0 -
                tic0 - frame_i->second.R.transpose() * dt * dt / 2.0 * g0;

            tmp_A.block<3, 3>(3, 0) = -Mat33::Identity();
            tmp_A.block<3, 3>(3, 3) =
                frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) =
                frame_i->second.R.transpose() * dt * Mat33::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) =
                frame_j->second.pre_integration->GetDeltaV() -
                frame_i->second.R.transpose() * dt * Mat33::Identity() * g0;

            Eigen::Matrix<double, 6, 6> cov_inv;
            cov_inv.setIdentity();

            MatXX r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VecX r_b  = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }

        A = A * 1000.0;
        b = b * 1000.0;

        x = A.ldlt().solve(b);

        VecX dg = x.segment<2>(n_state - 3);

        g0 = (g0 + lxly * dg).normalized() * G_const.norm();
    }

    g = g0;
}

bool LinearAlignment(std::map<double, ImageIMUFrame> &all_image_frame,
                     Vec3 &g,
                     VecX &x,
                     const Vec3 &G_const,
                     const Vec3 &tic0) {
    size_t all_frame_count = all_image_frame.size();
    size_t n_state         = all_frame_count * 3 + 3 + 1;

    MatXX A(n_state, n_state);
    A.setZero();

    VecX b(n_state);
    b.setZero();

    std::map<double, ImageIMUFrame>::iterator frame_i;
    std::map<double, ImageIMUFrame>::iterator frame_j;

    int i = 0;
    for (frame_i = all_image_frame.begin();
         std::next(frame_i) != all_image_frame.end();
         ++frame_i, ++i) {
        frame_j = std::next(frame_i);

        MatXX tmp_A(6, 10);
        tmp_A.setZero();

        VecX tmp_b(6);
        tmp_b.setZero();

        const double dt = frame_j->second.pre_integration->GetSumDeltaTime();

        tmp_A.block<3, 3>(0, 0) = -dt * Mat33::Identity();
        tmp_A.block<3, 3>(0, 6) =
            frame_i->second.R.transpose() * dt * dt / 2 * Mat33::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() *
                                  (frame_j->second.T - frame_i->second.T) /
                                  100.0;
        tmp_b.block<3, 1>(0, 0) =
            frame_j->second.pre_integration->GetDeltaP() +
            frame_i->second.R.transpose() * frame_j->second.R * tic0 - tic0;

        tmp_A.block<3, 3>(3, 0) = -Mat33::Identity();
        tmp_A.block<3, 3>(3, 3) =
            frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) =
            frame_i->second.R.transpose() * dt * Mat33::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->GetDeltaV();

        Eigen::Matrix<double, 6, 6> cov_inv;
        cov_inv.setIdentity();

        MatXX r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VecX r_b  = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }

    A = A * 1000.0;
    b = b * 1000.0;

    x = A.ldlt().solve(b);

    double s = x(n_state - 1) / 100.0;

    g = x.segment<3>(n_state - 4);

    if (std::fabs(g.norm() - G_const.norm()) > 1.0 || s < 0.0) {
        LOGW("SimpleVIO",
             "g.norm [%lf], G_const.norm [%lf], s [%lf]",
             g.norm(),
             G_const.norm(),
             s);
        return false;
    }

    RefineGravity(all_image_frame, g, x, G_const, tic0);

    s = (x.tail<1>())(0) / 100.0;

    (x.tail<1>())(0) = s;

    if (s < 0.0) {
        return false;
    } else {
        return true;
    }
}

bool VisualIMUAlignment(const size_t window_size,
                        std::map<double, ImageIMUFrame> &all_image_frame,
                        std::vector<Vec3> &Bgs,
                        Vec3 &g,
                        VecX &x,
                        const Vec3 &G_const,
                        const Vec3 &tic0) {
    SolveGyroscopeBias(window_size, all_image_frame, Bgs);

    if (LinearAlignment(all_image_frame, g, x, G_const, tic0)) {
        return true;
    } else {
        return false;
    }
}

}  // namespace SimpleVIO