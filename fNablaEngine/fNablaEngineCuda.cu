#include "fNablaEngineCuda.cuh"

#ifdef __INTELLISENSE__
#pragma diag_suppress 20, 29 //supress rhypot and kernel call errors
#endif

#define gpuCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			exit(code);
		} else {
			throw std::exception(cudaGetErrorString(code));
		}
	}
}

__global__ void fNablaEngineCuda::_AO_kernel(
	double* displacement,
	double3* normal,
	double2* directions,
	double* output,
	int rows,
	int cols,
	int step_displacement,
	int step_normal,
	int step_out,
	const int samples,
	const double radius,
	const double depth
) {
	const int i = blockIdx.y * blockDim.y + threadIdx.y; //2D coodinates of current thread
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i < rows) && (j < cols)) //Only valid threads perform memory I/O
	{
		const double displacement_ij = displacement[i * step_displacement + j];
		double3 normal_ij = normal[i * step_normal + j];
		double ao_ij = 0.0;

		for (int u = 0; u < samples; u++) {

			//double theta = M_PI * (((double)u) / (double)samples) - M_PI_2; //-pi/2 to pi/2
			const double sin_theta = directions[u].x; // sin(theta);
			const double cos_theta = directions[u].y; // cos(theta);

			//D={ sin(theta), cos(theta), 0 }; V={0, 0, 1}; SN = cross(D, V)
			//N = normal_ij; PN = N - SN * NdotSN = {sin_theta * NdotSN, cos_theta * NdotSN, normal_ij.x}

			double NdotSN = normal_ij.y * cos_theta + normal_ij.z * sin_theta;
			double rproj_length = rhypot(NdotSN, normal_ij.x);
			double n = atan2(normal_ij.x, NdotSN);
			double cos_n = NdotSN * rproj_length;
			double sin_n = normal_ij.x * rproj_length;

			double max_left = 0.0;
			double max_right = 0.0;

			//16 line samples per radial sample; 32 radial => 512 line which for up to 4k with the average of 0.25 distance needed means we are only skipping 1 pixel
			double step_size = radius / (samples * 16);
			//for (double length = 1.0; length <= radius; length++) {
			for (double length = step_size; length <= radius; length += step_size) {
				int j_step = lrint(sin_theta * length);
				int i_step = lrint(cos_theta * length);

				{
					int left_i = i + i_step;
					int left_j = j + j_step;

					//edge repeat
					//int left_i = max(min(left_i, rows-1), 0);
					//int left_j = max(min(left_j, cols-1), 0);
					//mirror
					left_i += rows * ((left_i < 0) - (left_i >= rows));
					left_j += cols * ((left_j < 0) - (left_j >= cols));

					double diff_left = (displacement[left_i * step_displacement + left_j] - displacement_ij);

					max_left = fmax(max_left, diff_left / length);
				}

				{
					int right_i = i - i_step;
					int right_j = j - j_step;

					//edge repeat
					//right_i = max(min(right_i, rows-1), 0);
					//right_j = max(min(right_j, cols-1), 0);
					//mirror
					right_i += rows * ((right_i < 0) - (right_i >= rows));
					right_j += cols * ((right_j < 0) - (right_j >= cols));

					double diff_right = (displacement[right_i * step_displacement + right_j] - displacement_ij);

					max_right = fmax(max_right, diff_right / length);
				}
			}
			double l = atan(depth * max_left);
			double r = atan(depth * max_right);

			//clamp to normal hemisphere
			l = n + fmin(l - n, _PI_2);
			r = n + fmin(r - n, _PI_2);

			//simplified formula that's very close:
			//ao_ij += normal_ij.x * (_PI - l - r) - hypot(NdotSN, normal_ij.x) + 1.0;
			ao_ij += (0.25 * cos(n + l + l) + 0.25 * cos(n + r + r) + 1.0 * (_PI - l - r) * sin_n + 0.5 * cos_n - 1.0) / rproj_length + 1.0;
		}

		output[i * step_out + j] = ao_ij / samples;
	}
}

void fNablaEngineCuda::ComputeAOCuda(cv::Mat& displacement, cv::Mat& normal, cv::Mat& ao, const int samples, const double distance, const double depth) {
	int cudaDevices;
	gpuCall(cudaGetDeviceCount(&cudaDevices));
	if (cudaDevices) {
		const cv::Size shape = cv::Size(ao.cols, ao.rows);
		const double max_size = double(std::min(shape.height, shape.width));

		const int displacementBytes = displacement.step * shape.height;
		double* gpu_displacement_data;
		gpuCall(cudaMalloc<double>(&gpu_displacement_data, displacementBytes));
		gpuCall(cudaMemcpy(gpu_displacement_data, displacement.data, displacementBytes, cudaMemcpyHostToDevice));

		const int normalBytes = normal.step * shape.height;
		double3* gpu_normal_data;
		gpuCall(cudaMalloc<double3>(&gpu_normal_data, normalBytes));
		gpuCall(cudaMemcpy(gpu_normal_data, normal.data, normalBytes, cudaMemcpyHostToDevice));


		cv::Mat directions(samples, 1, CV_64FC2);
		directions.forEach<cv::Point2d>([&](cv::Point2d& dir, const int* pos) -> void {
			double theta = 2.0 * _PI * ((double)pos[0] / (double)samples) - 2.0 * _PI_2; // -pi/2 to pi/2
			dir.x = sin(theta);
			dir.y = cos(theta);
		});
		const int directionsBytes = directions.step * directions.rows;
		double2* gpu_directions_data;
		gpuCall(cudaMalloc<double2>(&gpu_directions_data, directionsBytes));
		gpuCall(cudaMemcpy(gpu_directions_data, directions.data, directionsBytes, cudaMemcpyHostToDevice));

		const int outBytes = ao.step * shape.height;
		double* gpu_output_data;
		gpuCall(cudaMalloc<double>(&gpu_output_data, outBytes));

		const dim3 block(16, 16);
		const dim3 grid((shape.width + block.x - 1) / block.x, (shape.height + block.y - 1) / block.y);

		fNablaEngineCuda::_AO_kernel<<<grid, block>>>(
			gpu_displacement_data,
			gpu_normal_data,
			gpu_directions_data,
			gpu_output_data,
			shape.height,
			shape.width,
			displacement.step / sizeof(double),
			normal.step / sizeof(double3),
			ao.step / sizeof(double),
			samples,
			std::max(distance * max_size, 1.0),
			depth * max_size
		);
		gpuCall(cudaPeekAtLastError());

		gpuCall(cudaDeviceSynchronize());

		gpuCall(cudaMemcpy(ao.data, gpu_output_data, outBytes, cudaMemcpyDeviceToHost));

		gpuCall(cudaFree(gpu_displacement_data));
		gpuCall(cudaFree(gpu_normal_data));
		gpuCall(cudaFree(gpu_directions_data));
		gpuCall(cudaFree(gpu_output_data));
	}
}