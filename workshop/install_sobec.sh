git clone --recursive https://github.com/MeMory-of-MOtion/sobec.git
pip install ndcurves
cd sobec
git checkout -b topic/wan24
export CMAKE_PREFIX_PATH="$(cmeel cmake)"
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$(cmeel cmake)" -DCMAKE_BUILD_TYPE=RELEASE
