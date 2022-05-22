#include "problem.h"

#include "problem_DogLeg.h"
#include "problem_LM_Marquardt.h"
#include "problem_LM_Nielson.h"

#include "utils/logger.h"

namespace SimpleVIO {

ProblemPtr CreateProblemPtr(Problem::OptimizationMethod opt_method,
                            Problem::ProblemType problem_type) {
    switch (opt_method) {
        case SimpleVIO::Problem::OptimizationMethod::LM_NIELSON: {
            return std::make_shared<Problem_LM_Nielson>(problem_type);
        }
        case SimpleVIO::Problem::OptimizationMethod::LM_MARQUARDT: {
            return std::make_shared<Problem_LM_Marquardt>(problem_type);
        }
        case SimpleVIO::Problem::OptimizationMethod::DOGLEG: {
            return std::make_shared<Problem_DogLeg>(problem_type);
        }
        default: {
            LOGE("SimpleVIO", "unsupport optimization method %d", opt_method);

            break;
        }
    }

    return nullptr;
}

}  // namespace SimpleVIO