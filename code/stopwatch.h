/**
 * stopwatch.h: Provides a simple stopwatch to measure the execution time of a program or a part of it.
 *
 * Usage:
 * ```
 * StopWatch stopwatch; // buy a new stopwatch
 * 
 * stopwatch.start(); // reset to zero and start
 * // execute some time-consuming operations
 * double duration = stopwatch.lap(); // get the lap time
 * std::cout << "LapTime: " << duration << " seconds" << std::endl;
 * 
 * // execute another time-consuming operations
 * duration = stopwatch.stop(); // get the time and reset to zero
 * std::cout << "TotalTime: " << duration << " seconds" << std::endl;
 * ```
 */

#ifndef STOPWATCH_H
#define STOPWATCH_H
#include <chrono>


class StopWatch{
    private:
        std::chrono::system_clock::time_point start_;
    public:
        StopWatch() : start_(std::chrono::system_clock::time_point::min()) {}
        void start(){
            start_ = std::chrono::system_clock::now();
        }
        double lap(){
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            double duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000000.0);
            return duration;
        }
        double stop(){
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            double duration = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000000.0);
            start_ = std::chrono::system_clock::time_point::min();
            return duration;
        }
};


#endif // STOPWATCH_H
