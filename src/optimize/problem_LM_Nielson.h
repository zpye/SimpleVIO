#ifndef SIMPLE_VIO_OPTIMIZE_PROBLEM_LM_NIELSON_H_
#define SIMPLE_VIO_OPTIMIZE_PROBLEM_LM_NIELSON_H_

#include "problem.h"

namespace SimpleVIO {

class Problem_LM_Nielson : public Problem {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Problem_LM_Nielson(ProblemType problem_type) : Problem(problem_type) {}

public:
    virtual bool Solve(int iterations = 10) override;

private:
    void ComputeParamsInit();

    void SolveLinearSystem(bool use_shur_marg);

    bool IsGoodDelta();

    bool IsGoodStep();

private:
    struct ParamsLM {
        double current_lambda    = -1.0;
        double current_chi2      = -1.0;
        double stop_threshold_LM = -1.0;
        double ni                = -1.0;
    };
    ParamsLM params_LM_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_OPTIMIZE_PROBLEM_LM_NIELSON_H_