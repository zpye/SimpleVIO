#include "solve_relative_pose.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace SimpleVIO {

void InverseRT(Mat33 &R, Vec3 &T) {
    Mat33 tempR = R.transpose();
    Vec3 tempT = -tempR * T;
    R = tempR;
    T = tempT;
}

bool SolveRelativeRT(const std::vector<std::pair<Vec3, Vec3>> &corres,
                     Mat33 &R,
                     Vec3 &T,
                     const int min_inlier_size,
                     const size_t min_corres_size,
                     const double ransac_threshold,
                     const double ransac_confidence) {
    if (corres.size() >= min_corres_size) {
        std::vector<cv::Point2f> left_pts;
        std::vector<cv::Point2f> right_pts;

        for (auto &lr : corres) {
            left_pts.emplace_back(lr.first(0), lr.first(1));
            right_pts.emplace_back(lr.second(0), lr.second(1));
        }

        cv::Mat mask;

        // TODO: set RANSAC parameters from outside
        cv::Mat E = cv::findFundamentalMat(left_pts,
                                           right_pts,
                                           cv::FM_RANSAC,
                                           ransac_threshold,
                                           ransac_confidence,
                                           mask);

        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 1.0,
                                 0.0,
                                 0.0,
                                 0.0,
                                 1.0);

        cv::Mat rotation;
        cv::Mat translation;
        int inlier_cnt = cv::recoverPose(E,
                                         left_pts,
                                         right_pts,
                                         camera_matrix,
                                         rotation,
                                         translation,
                                         mask);

        cv::cv2eigen(rotation, R);
        cv::cv2eigen(translation, T);

        if (inlier_cnt > min_inlier_size) {
            return true;
        } else {
            return false;
        }
    }

    R = Mat33::Identity();
    T = Vec3::Zero();

    return false;
}

bool SolveRelativeR(const std::vector<std::pair<Vec3, Vec3>> &corres,
                    Mat33 &R,
                    const int min_inlier_size,
                    const size_t min_corres_size,
                    const double ransac_threshold,
                    const double ransac_confidence) {
    Vec3 T;  // not used
    return SolveRelativeRT(corres,
                           R,
                           T,
                           min_inlier_size,
                           min_corres_size,
                           ransac_threshold,
                           ransac_confidence);
}

}  // namespace SimpleVIO
