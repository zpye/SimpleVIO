#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>

#include "backend/edge_prior.h"
#include "backend/edge_reprojection.h"
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "optimize/problem.h"

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

using namespace SimpleVIO;

struct Frame {
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t){};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    std::unordered_map<int, Eigen::Vector3d> featurePerId;
};

void GetSimDataInWordFrame(std::vector<Frame> &cameraPoses,
                           std::vector<Eigen::Vector3d> &points) {
    int featureNums = 20;
    int poseNums    = 3;

    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4);

        Eigen::Matrix3d R;
        R                 = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius,
                                            radius * sin(theta),
                                            1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.0);
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4.0, 4.0);
        std::uniform_real_distribution<double> z_rand(4.0, 8.0);

        Eigen::Vector3d Pw(xy_rand(generator),
                           xy_rand(generator),
                           z_rand(generator));
        points.push_back(Pw);

        for (int i = 0; i < poseNums; ++i) {
            Eigen::Vector3d Pc =
                cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            Pc = Pc / Pc.z();
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(std::make_pair(j, Pc));
        }
    }
}

int main() {
    std::vector<Frame> cameras;
    std::vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points);
    Eigen::Quaterniond qic(1, 0, 0, 0);
    Eigen::Vector3d tic(0, 0, 0);

    std::shared_ptr<VertexPose> vertex_extrinsic(new VertexPose());
    Eigen::VectorXd ex_pose(7);
    ex_pose << tic, qic.x(), qic.y(), qic.z(), qic.w();
    vertex_extrinsic->SetParameters(ex_pose.data());
    vertex_extrinsic->SetFixed(true);

    ProblemPtr problem =
        CreateProblemPtr(Problem::OptimizationMethod::LM_NIELSON,
                         Problem::ProblemType::SLAM_PROBLEM);

    std::vector<std::shared_ptr<VertexPose> > vertexCams_vec;
    for (size_t i = 0; i < cameras.size(); ++i) {
        std::shared_ptr<VertexPose> vertexCam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(),
            cameras[i].qwc.z(), cameras[i].qwc.w();
        vertexCam->SetParameters(pose.data());

        // if (i < 2) {
        //     vertexCam->SetFixed();
        // }

        if (i < 2) {
            std::shared_ptr<EdgeSE3Prior> edge_prior(
                new EdgeSE3Prior(cameras[i].twc, cameras[i].qwc));
            edge_prior->SetVertex(0, vertexCam);

            int residual_dim = edge_prior->GetResidualDimention();
            Eigen::MatrixXd information(residual_dim, residual_dim);
            information.setIdentity();
            information = information * 0.01;
            edge_prior->SetInformation(information.data());

            problem->AddEdge(edge_prior);
        }

        problem->AddVertex(vertexCam);
        vertexCams_vec.push_back(vertexCam);
    }

    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0, 1.);
    double noise = 0;
    std::vector<double> noise_invd;
    std::vector<std::shared_ptr<VertexInverseDepth> > allPoints;
    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector3d Pw = points[i];
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc);
        noise              = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise);
        noise_invd.push_back(inverse_depth);

        std::shared_ptr<VertexInverseDepth> verterxPoint(
            new VertexInverseDepth());
        VecX inv_d(1);
        inv_d << inverse_depth;
        verterxPoint->SetParameters(inv_d.data());
        problem->AddVertex(verterxPoint);
        allPoints.push_back(verterxPoint);

        for (size_t j = 1; j < cameras.size(); ++j) {
            Eigen::Vector3d pt_i = cameras[0].featurePerId.find(i)->second;
            Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
            std::shared_ptr<EdgeReprojection> edge(
                new EdgeReprojection(pt_i, pt_j));

            edge->SetVertex(0, verterxPoint);
            edge->SetVertex(1, vertexCams_vec[0]);
            edge->SetVertex(2, vertexCams_vec[j]);
            edge->SetVertex(3, vertex_extrinsic);

            problem->AddEdge(edge);
        }
    }

    problem->Solve(5);

    std::cout << "\nCompare MonoBA results after opt..." << std::endl;
    for (size_t k = 0; k < allPoints.size(); k += 1) {
        double inv_depth[1];
        allPoints[k]->GetParameters(inv_depth);
        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z()
                  << " ,noise " << noise_invd[k] << " ,opt " << inv_depth[0]
                  << std::endl;
    }
    std::cout << "------------ pose translation ----------------" << std::endl;
    for (int i = 0; i < vertexCams_vec.size(); ++i) {
        VecX pose(7);
        vertexCams_vec[i]->GetParameters(pose.data());
        std::cout << "translation after opt: " << i << " :"
                  << pose.head(3).transpose()
                  << " || gt: " << cameras[i].twc.transpose() << std::endl;
    }

    return 0;
}
