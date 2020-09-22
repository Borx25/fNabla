#pragma once

//cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Opencv
#include <opencv2/core/core.hpp>

//CV_TYPE
//+ -------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +
//|channels:|	1	|	2	|	3	|	4	|	5	|	6	|	7	|	8	|
//+-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +
//|	CV_8U	|	0	|	8	|	16	|	24	|	32	|	40	|	48	|	56	|
//|	CV_8S	|	1	|	9	|	17	|	25	|	33	|	41	|	49	|	57	|
//|	CV_16U	|	2	|	10	|	18	|	26	|	34	|	42	|	50	|	58	|
//|	CV_16S	|	3	|	11	|	19	|	27	|	35	|	43	|	51	|	59	|
//|	CV_32S	|	4	|	12	|	20	|	28	|	36	|	44	|	52	|	60	|
//|	CV_32F	|	5	|	13	|	21	|	29	|	37	|	45	|	53	|	61	|
//|	CV_64F	|	6	|	14	|	22	|	30	|	38	|	46	|	54	|	62	|
//+-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +


#include "fNablaConstants.h"

namespace fNablaEngineCuda
{
#define CUDA_ERROR_CHECK(code) { _gpuErrorCheck((code), __FILE__, __LINE__); }
	class NoCUDAGPU : std::runtime_error {
	public:
		NoCUDAGPU() : std::runtime_error("No CUDA-capable GPU found") { }
	};
	inline void  _gpuErrorCheck(cudaError_t code, const char* file, int line);
	//check if a CUDA-capable gpu is present, otherwise throws a NoCUDAGPU exception
	void CheckCUDACapable();

	template<typename T>
	class GPUMat {
		size_t bytes;
	public:
		T* data;

		void Download(cv::Mat& output) {
			CUDA_ERROR_CHECK(cudaMemcpy(output.data, data, bytes, cudaMemcpyDeviceToHost));
		}

		GPUMat(cv::Mat& input) {
			bytes = input.total() * input.elemSize();
			CUDA_ERROR_CHECK(cudaMalloc<T>(&data, bytes));
			CUDA_ERROR_CHECK(cudaMemcpy(data, input.data, bytes, cudaMemcpyHostToDevice));
		}

		GPUMat(const dim3 shape) {
			bytes = shape.x * shape.y * shape.z * sizeof(T);
			CUDA_ERROR_CHECK(cudaMalloc<T>(&data, bytes));
		}

		~GPUMat() {
			CUDA_ERROR_CHECK(cudaFree(data));
		}
	};

#define CUDA_LAUNCH(KERNEL, GRID, BLOCK, ...) KERNEL<<<GRID, BLOCK>>>(__VA_ARGS__)

	__global__ void _pre_kernel(
		double2* directions,
		const uint samples
	);

	__device__ __forceinline__ double IntegrateArc(double2 h, double n, double sin_n, double cos_n);

	enum class Edge_mode {
		Tile,
		Repeat
	};

	__device__ __forceinline__ void TestHorizon(
		double* displacement,
		dim3 shape,
		int2 pos,
		double distance,
		double height_ref,
		double& horizon,
		Edge_mode mode = Edge_mode::Tile,
		bool invert = false
	);

	__global__ void _AO_kernel(
		double* displacement,
		double3* normal,
		double2* directions,
		double* output,
		dim3 shape,
		const double radius,
		const double step_size,
		const double depth
	);

	__global__ void reduce_kernel(
		double* input,
		double* output,
		const dim3 shape
	);

	//Calculates ambient occlusion with a horizon-based algorithm
	void ComputeAOCuda(
		cv::Mat& displacement,
		cv::Mat& normal,
		cv::Mat& ao,
		const uint samples,
		const double distance,
		const double depth,
		std::string& status
	);
}
