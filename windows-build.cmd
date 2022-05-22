mkdir build

cd build

set VCPKG_ROOT=E:/codes/vcpkg

cmake .. ^
    -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake ^
    -DDRAW_RESULT=ON ^
    -DUSE_OPENMP=ON