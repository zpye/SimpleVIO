#include <iostream>
#include <memory>
#include <random>

#include "optimize//vertex.h"
#include "optimize/edge.h"
#include "optimize/problem.h"

#include "utils//eigen_types.h"

using namespace SimpleVIO;

class CurveFittingVertex : public Vertex<3, 3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingVertex() {}

    virtual ~CurveFittingVertex() {}

    virtual std::string TypeInfo() const override {
        return "abc";
    }
};

class CurveFittingEdge : public Edge<1, 1> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x, double y) {
        x_ = x;
        y_ = y;

        SetVerticesTypes({"abc"});
    }

    virtual void ComputeResidual() override {
        Vec3 abc;
        vertices_[0]->GetParameters(abc.data());
        residual_(0) = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2)) - y_;
    }

    virtual void ComputeJacobians() override {
        Vec3 abc;
        vertices_[0]->GetParameters(abc.data());
        double exp_y = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));

        Eigen::Matrix<double, 1, 3> jaco_abc;
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y, 1 * exp_y;
        jacobians_[0] = jaco_abc;
    }

    virtual std::string TypeInfo() const override {
        return "CurveFittingEdge";
    }

public:
    double x_, y_;
};

int main() {
    double a = 1.0, b = 2.0, c = 1.0;
    int N          = 200;
    double w_sigma = 1.0;

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0., w_sigma);

    ProblemPtr problem =
        CreateProblemPtr(Problem::OptimizationMethod::LM_MARQUARDT,
                         Problem::ProblemType::GENERIC_PROBLEM);

    std::shared_ptr<CurveFittingVertex> vertex(new CurveFittingVertex());

    double temp[3] = {0.0, 0.0, 0.0};
    vertex->SetParameters(temp);
    problem->AddVertex(vertex);

    for (int i = 0; i < N; ++i) {
        double x = i / 200.0;
        double n = noise(generator);
        double y = std::exp(a * x * x + b * x + c) + n;

        std::shared_ptr<CurveFittingEdge> edge(new CurveFittingEdge(x, y));
        edge->SetVertex(0, vertex);
        problem->AddEdge(edge);
    }

    std::cout << "\nTest CurveFitting start..." << std::endl;
    problem->Solve(30);

    std::cout << "-------After optimization, we got these parameters :"
              << std::endl;

    Vec3 result;
    vertex->GetParameters(result.data());
    std::cout << result.transpose() << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << "1.0,  2.0,  1.0" << std::endl;

    return 0;
}
