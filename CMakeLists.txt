CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(ceres_experiment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include 3rd libraries path
message("OS: ${CMAKE_SYSTEM_NAME}")

set(Ceres_DIR "/home/manuel/libraries/ceres/lib/cmake/Ceres")
set(glog_DIR "/home/manuel/libraries/glog/lib/cmake/glog")

# Glog
find_package(glog REQUIRED)

# Ceres
find_package(Ceres REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main 
  glog::glog
  Ceres::ceres
)
