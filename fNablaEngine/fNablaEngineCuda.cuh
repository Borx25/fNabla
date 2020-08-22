#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


constexpr double _PI = 3.14159265358979323846;
constexpr double _PI_2 = 1.57079632679489661923;
constexpr double _2_PI = 6.28318530717958647692;
constexpr double _PI_PI = 9.86960440108935861883;
constexpr double _4_PI_PI = 39.4784176043574344753;

namespace fNablaEngineCuda
{
	void ComputeAOCuda(
		cv::Mat& displacement,
		cv::Mat& normal,
		cv::Mat& ao,
		const int samples,
		const double distance,
		const double depth
	);

	__global__ void _AO_kernel(
		double* displacement,
		double3* normal,
		double2* directions,
		double* output,
		int width,
		int height,
		int step_displacement,
		int step_normal,
		int step_samples,
		int step_out,
		const int samples,
		const int radius,
		const double depth
	);
}