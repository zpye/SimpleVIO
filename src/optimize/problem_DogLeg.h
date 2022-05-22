#ifndef SIMPLE_VIO_OPTIMIZE_PROBLEM_DOGLEG_H_
#define SIMPLE_VIO_OPTIMIZE_PROBLEM_DOGLEG_H_

#include "problem.h"

namespace SimpleVIO {

class Problem_DogLeg : public Problem {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Problem_DogLeg(ProblemType problem_type) : Problem(problem_type) {}

public:
    virtual bool Solve(int iterations = 10) override;

private:
    void ComputeParamsInit();

    void SolveLinearSystem(bool use_shur_marg);

    bool IsGoodDelta();

    bool IsGoodStep();

private:
    void DogLegComputeXNorm();

    void DogLegComputeDeltaX();

    bool DogLegCondition1();

    bool DogLegCondition2_h_dl();

    bool DogLegCondition2_trust_region();

    bool DogLegUpdateTrustRegion();

private:
    struct ParamsDogLeg {
        double trust_region;
        double current_chi2;
        double rho;
        double norm_x;

        int state;
        double alpha = 0.0;
        double beta  = 1.0;
        VecX delta_x_gn;
        VecX delta_x_sd;

        double eps1;
        double eps2;
        double eps3;
    };
    ParamsDogLeg params_dogleg_;
};
}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_PROBLEM_DOGLEG_H_