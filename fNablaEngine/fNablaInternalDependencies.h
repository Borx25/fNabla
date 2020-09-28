#pragma once

//Opencv
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//MKL
#define USE_MKL

#ifdef USE_MKL
#include <mkl_dfti.h>
#endif // USE_MKL

//Engine components
#include "fNablaEngineCuda.cuh"
