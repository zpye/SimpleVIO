#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

static const int nDelayTimes = 2;
static std::string imu_data_file;
static std::string image_poses_data_file;
static std::string image_keyframes_data_dir;
static std::string config_file;

std::shared_ptr<SimpleVIO::System> pSystem;

void PubIMUData() {
    std::cout << "1 PubImuData start sImu_data_filea: " << imu_data_file
              << std::endl;
    std::ifstream fsImu;
    fsImu.open(imu_data_file.c_str());
    if (!fsImu.is_open()) {
        std::cerr << "Failed to open imu file! " << imu_data_file << std::endl;
        return;
    }

    std::string sImu_line;
    double dStampNSec = 0.0;
    Eigen::Vector3d vAcc;
    Eigen::Vector3d vGyr;

    // read imu data
    while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) {
        std::istringstream ssImuData(sImu_line);

        double no_used_pose[7];
        ssImuData >> dStampNSec >> no_used_pose[0] >> no_used_pose[1] >>
            no_used_pose[2] >> no_used_pose[3] >> no_used_pose[4] >>
            no_used_pose[5] >> no_used_pose[6] >> vGyr.x() >> vGyr.y() >>
            vGyr.z() >> vAcc.x() >> vAcc.y() >> vAcc.z();
        // cout << "Imu t: " << fixed << dStampNSec << " gyr: " <<
        // vGyr.transpose()
        // << " acc: " << vAcc.transpose() << endl;

        pSystem->PubIMUData(dStampNSec, vAcc, vGyr);

        std::this_thread::sleep_for(
            std::chrono::microseconds(5000 * nDelayTimes));
    }

    fsImu.close();
}

void PubImageData() {
    std::cout << "1 PubImageData start sImage_file: " << image_poses_data_file
              << std::endl;

    std::ifstream fsImage;
    fsImage.open(image_poses_data_file.c_str());
    if (!fsImage.is_open()) {
        std::cerr << "Failed to open image file! " << image_poses_data_file
                  << std::endl;
        return;
    }

    std::vector<double> dStampNSec_vec;

    std::string sImage_line;
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {
        std::istringstream image_line_data(sImage_line);
        double dStampNSec;
        image_line_data >> dStampNSec;

        dStampNSec_vec.push_back(dStampNSec);
    }

    printf("image pose size: %zu\n", dStampNSec_vec.size());
    for (std::size_t i = 0; i < dStampNSec_vec.size(); ++i) {
        // std::cout << "images: " << i << "/" << dStampNSec_vec.size()
        //           << std::endl;
        char file_name[256];
        sprintf(file_name,
                "%s/all_points_%d.txt",
                image_keyframes_data_dir.c_str(),
                (int)i);

        std::string keyframes_file(file_name);

        pSystem->PubImageData(dStampNSec_vec[i], keyframes_file);

        std::this_thread::sleep_for(
            std::chrono::microseconds(50000 * nDelayTimes));
    }

    fsImage.close();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "./run_generate_data PATH/TO/IMU_DATA PATH/TO/POSES "
                     "PATH/TO/IMAGE_DATA PATH/TO/CONFIG \n"
                  << std::endl;
        return -1;
    }

    imu_data_file            = argv[1];
    image_poses_data_file    = argv[2];
    image_keyframes_data_dir = argv[3];
    config_file              = argv[4];

    pSystem = std::make_shared<SimpleVIO::System>();
    // pSystem.reset(new System(config_file));
    pSystem->Initialize(config_file);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // sleep(5);
    std::thread thd_PubImuData(PubIMUData);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::thread thd_PubImageData(PubImageData);

    thd_PubImuData.join();
    thd_PubImageData.join();

    std::this_thread::sleep_for(std::chrono::seconds(15));

    std::cout << "system Finish" << std::endl;
    pSystem->Finish();

    std::cout << "main end... see you ..." << std::endl;

    return 0;
}
