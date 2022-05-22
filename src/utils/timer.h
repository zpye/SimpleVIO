#ifndef SIMPLE_VIO_UTILS_TIMER_H_
#define SIMPLE_VIO_UTILS_TIMER_H_

#include <chrono>

namespace SimpleVIO {

class Timer {
public:
    Timer() {
        Start();
    }

    void Start() {
        start_ = std::chrono::system_clock::now();
    }

    double End() {
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;

        return (elapsed_seconds.count() * 1000);
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_;
    std::chrono::time_point<std::chrono::system_clock> end_;
};

}  // namespace SimpleVIO

#endif  // SIMPLE_VIO_UTILS_TIMER_H_