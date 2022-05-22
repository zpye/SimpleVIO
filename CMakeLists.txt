cmake_minimum_required(VERSION 3.0)

project(simple_vio)

option(DRAW_RESULT "draw points and trajectories" ON)
option(USE_OPENMP "use openmp to accelerate" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /EHsc")
endif()

find_package(Threads REQUIRED)

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# 3rdparty
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)

message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")

if(DRAW_RESULT)
  find_package(Pangolin REQUIRED)
  add_definitions(-DDRAW_RESULT)
endif()

set(VIO_INC_DIRS ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS}
                 ${Pangolin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
message(STATUS "VIO_INC_DIRS: ${VIO_INC_DIRS}")

set(VIO_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
message(STATUS "VIO_SRC_DIR: ${VIO_SRC_DIR}")

# camera models
add_library(
  camera_model STATIC
  ${VIO_SRC_DIR}/camodocal/chessboard/Chessboard.cc
  ${VIO_SRC_DIR}/camodocal/calib/CameraCalibration.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/Camera.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/CameraFactory.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/CostFunctionFactory.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/PinholeCamera.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/CataCamera.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/EquidistantCamera.cc
  ${VIO_SRC_DIR}/camodocal/camera_models/ScaramuzzaCamera.cc
  ${VIO_SRC_DIR}/camodocal/sparse_graph/Transform.cc
  ${VIO_SRC_DIR}/camodocal/gpl/gpl.cc
  ${VIO_SRC_DIR}/camodocal/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(camera_model PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS}
                                          ${CERES_LIBRARIES})

target_include_directories(camera_model PUBLIC ${VIO_INC_DIRS} ${VIO_SRC_DIR})

# optimize library
add_library(
  optimize STATIC
  ${VIO_SRC_DIR}/optimize/base_vertex.h
  ${VIO_SRC_DIR}/optimize/base_vertex.cpp
  ${VIO_SRC_DIR}/optimize/base_edge.h
  ${VIO_SRC_DIR}/optimize/base_edge.cpp
  ${VIO_SRC_DIR}/optimize/vertex.h
  ${VIO_SRC_DIR}/optimize/edge.h
  ${VIO_SRC_DIR}/optimize/loss_function.h
  ${VIO_SRC_DIR}/optimize/loss_function.cpp
  ${VIO_SRC_DIR}/optimize/problem.h
  ${VIO_SRC_DIR}/optimize/problem.cpp
  ${VIO_SRC_DIR}/optimize/problem_manager.cpp
  ${VIO_SRC_DIR}/optimize/problem_LM_Nielson.h
  ${VIO_SRC_DIR}/optimize/problem_LM_Nielson.cpp
  ${VIO_SRC_DIR}/optimize/problem_LM_Marquardt.h
  ${VIO_SRC_DIR}/optimize/problem_LM_Marquardt.cpp
  ${VIO_SRC_DIR}/optimize/problem_DogLeg.h
  ${VIO_SRC_DIR}/optimize/problem_DogLeg.cpp)

target_include_directories(optimize PUBLIC ${VIO_INC_DIRS}
                           ${VIO_SRC_DIR} ${VIO_SRC_DIR}/3rdparty)

if(USE_OPENMP)
  target_link_libraries(optimize PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(optimize PUBLIC -DUSE_OPENMP)
endif()

# core library
add_library(
  SimpleVIO STATIC
  ${VIO_SRC_DIR}/System.h
  ${VIO_SRC_DIR}/System.cpp
  ${VIO_SRC_DIR}/parameters.h
  ${VIO_SRC_DIR}/parameters.cpp
  ${VIO_SRC_DIR}/estimator/estimator.h
  ${VIO_SRC_DIR}/estimator/estimator.cpp
  ${VIO_SRC_DIR}/feature/feature_manager.h
  ${VIO_SRC_DIR}/feature/feature_manager.cpp
  ${VIO_SRC_DIR}/feature/feature_tracker.h
  ${VIO_SRC_DIR}/feature/feature_tracker.cpp
  ${VIO_SRC_DIR}/initialize/solve_relative_pose.h
  ${VIO_SRC_DIR}/initialize/solve_relative_pose.cpp
  ${VIO_SRC_DIR}/initialize/initial_alignment.h
  ${VIO_SRC_DIR}/initialize/initial_alignment.cpp
  ${VIO_SRC_DIR}/initialize/global_sfm.h
  ${VIO_SRC_DIR}/initialize/global_sfm.cpp
  ${VIO_SRC_DIR}/initialize/initial_ex_rotation.h
  ${VIO_SRC_DIR}/initialize/initial_ex_rotation.cpp
  ${VIO_SRC_DIR}/imu/imu_utils.h
  ${VIO_SRC_DIR}/imu/imu_integration.h
  ${VIO_SRC_DIR}/imu/imu_integration.cpp
  ${VIO_SRC_DIR}/backend/vertex_pose.h
  ${VIO_SRC_DIR}/backend/vertex_pose.cpp
  ${VIO_SRC_DIR}/backend/vertex_point_xyz.h
  ${VIO_SRC_DIR}/backend/vertex_inverse_depth.h
  ${VIO_SRC_DIR}/backend/vertex_speed_bias.h
  ${VIO_SRC_DIR}/backend/edge_imu.h
  ${VIO_SRC_DIR}/backend/edge_imu.cpp
  ${VIO_SRC_DIR}/backend/edge_prior.h
  ${VIO_SRC_DIR}/backend/edge_prior.cpp
  ${VIO_SRC_DIR}/backend/edge_reprojection.h
  ${VIO_SRC_DIR}/backend/edge_reprojection.cpp
  ${VIO_SRC_DIR}/utils/eigen_types.h
  ${VIO_SRC_DIR}/utils/rotation_utils.h
  ${VIO_SRC_DIR}/utils/logger.h
  ${VIO_SRC_DIR}/utils/timer.h)

target_link_libraries(
  SimpleVIO PUBLIC ${OpenCV_LIBS} ${CERES_LIBRARIES} ${Pangolin_LIBRARIES}
                   camera_model optimize Threads::Threads)

target_include_directories(SimpleVIO PUBLIC ${VIO_INC_DIRS}
                           ${VIO_SRC_DIR} ${VIO_SRC_DIR}/3rdparty)

# tests
add_executable(run_euroc ${CMAKE_SOURCE_DIR}/test/run_euroc.cpp)
target_link_libraries(run_euroc PUBLIC SimpleVIO)

add_executable(run_generate_data ${CMAKE_SOURCE_DIR}/test/run_generate_data.cpp)
target_link_libraries(run_generate_data PUBLIC SimpleVIO)

add_executable(test_curve_fitting ${CMAKE_SOURCE_DIR}/test/test_curve_fitting.cpp)
target_link_libraries(test_curve_fitting PUBLIC optimize)

add_executable(test_mono_ba ${CMAKE_SOURCE_DIR}/test/test_mono_ba.cpp
                           ${VIO_SRC_DIR}/backend/vertex_pose.h
                           ${VIO_SRC_DIR}/backend/vertex_pose.cpp
                           ${VIO_SRC_DIR}/backend/vertex_inverse_depth.h
                           ${VIO_SRC_DIR}/backend/edge_prior.h
                           ${VIO_SRC_DIR}/backend/edge_prior.cpp
                           ${VIO_SRC_DIR}/backend/edge_reprojection.h
                           ${VIO_SRC_DIR}/backend/edge_reprojection.cpp)
target_link_libraries(test_mono_ba PUBLIC optimize)
