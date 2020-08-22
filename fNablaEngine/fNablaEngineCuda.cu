#include "fNablaEngineCuda.cuh"

#ifdef __INTELLISENSE__
#pragma diag_suppress 20, 29 //supress rhypot and kernel call errors
#endif

__global__ void fNablaEngineCuda::_AO_kernel(
	double* displacement,
	double3* normal,
	double2* directions,
	double* output,
	int rows,
	int cols,
	int step_displacement,
	int step_normal,
	int step_directions,
	int step_out,
	const int samples,
	const int radius,
	const double depth
) {
	const int i = blockIdx.y * blockDim.y + threadIdx.y; //2D coodinates of current thread
	const int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i < rows) && (j < cols)) //Only valid threads perform memory I/O
	{
		const double displacement_ij = displacement[i * step_displacement + j];
		const double3 normal_ij = normal[i * step_normal + j];
		double ao_ij = 0.0;

		for (int u = 0; u < samples; u++) {

			//double theta = M_PI * (((double)u) / (double)samples) - M_PI_2; //-pi/2 to pi/2
			const double sin_theta = directions[u * step_directions].x; // sin(theta);
			const double cos_theta = directions[u * step_directions].y; // cos(theta);

			//D={ sin(theta), cos(theta), 0 }; V={0, 0, 1}; SN = cross(D, V)
			//N = normal_ij; PN = N - SN * NdotSN = {sin_theta * NdotSN, cos_theta * NdotSN, normal_ij.x}

			double NdotSN = normal_ij.y * cos_theta + normal_ij.z * sin_theta;
			double rproj_length = rhypot(NdotSN, normal_ij.x);
			double n = atan2(normal_ij.x, NdotSN);
			double cos_n = NdotSN * rproj_length;
			double sin_n = normal_ij.x * rproj_length;

			double max_left = 0.0;
			double max_right = 0.0;

			for (double length = 1.0; length <= radius; length++) {
				int j_step = lrint(sin_theta * length);
				int i_step = lrint(cos_theta * length);

				{
					int left_i = i + i_step;
					int left_j = j + j_step;
					//int left_i = i + step_vector_int.y;
					//int left_j = j + step_vector_int.x;
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

			ao_ij += (0.25 * cos(n + l + l) + 0.25 * cos(n + r + r) + 0.5 * (-r - l + _PI) * sin_n + 0.5 * cos_n - 1.0) / rproj_length + 1.0;
		}

		output[i * step_out + j] = ao_ij / samples;
	}
}

void fNablaEngineCuda::ComputeAOCuda(cv::Mat& displacement, cv::Mat& normal, cv::Mat& ao, const int samples, const double distance, const double depth) {
	int cudaDevices;
	cudaGetDeviceCount(&cudaDevices);
	if (cudaDevices)
	{
		const cv::Size shape = cv::Size(ao.cols, ao.rows);

		const double max_size = double(std::min(shape.height, shape.width));
		const int radius = std::max(int(distance * max_size), 1);

		const cv::Mat directions(samples, radius, CV_64FC2);
		directions.forEach<cv::Point2d>([&](cv::Point2d& dir, const int* pos) -> void {
			double theta = 2.0 * _PI * ((double)pos[0] / (double)samples) - 2.0 * _PI_2; // -pi/2 to pi/2
			double mult = double(pos[1]) + 1.0;
			dir.x = sin(theta) * mult;
			dir.y = cos(theta) * mult;
			});

		const int displacementBytes = displacement.step * shape.height;
		double* gpu_displacement_data;
		cudaMalloc<double>(&gpu_displacement_data, displacementBytes);
		cudaMemcpy(gpu_displacement_data, displacement.data, displacementBytes, cudaMemcpyHostToDevice);
		int step_displacement = displacement.step / sizeof(double);

		const int normalBytes = normal.step * shape.height;
		double3* gpu_normal_data;
		cudaMalloc<double3>(&gpu_normal_data, normalBytes);
		cudaMemcpy(gpu_normal_data, normal.data, normalBytes, cudaMemcpyHostToDevice);
		int step_normal = normal.step / sizeof(double3);

		const int directionsBytes = directions.step * directions.rows;
		double2* gpu_directions_data;
		cudaMalloc<double2>(&gpu_directions_data, directionsBytes);
		cudaMemcpy(gpu_directions_data, directions.data, directionsBytes, cudaMemcpyHostToDevice);
		int step_directions = directions.step / sizeof(double2);

		const int outBytes = ao.step * shape.height;
		double* gpu_output_data;
		cudaMalloc<double>(&gpu_output_data, outBytes);
		int step_output = ao.step / sizeof(double);

		const dim3 block(16, 16);
		const dim3 grid((shape.width + block.x - 1) / block.x, (shape.height + block.y - 1) / block.y);

		fNablaEngineCuda::_AO_kernel << <grid, block >> > (
			gpu_displacement_data,
			gpu_normal_data,
			gpu_directions_data,
			gpu_output_data,
			shape.height,
			shape.width,
			step_displacement,
			step_normal,
			step_directions,
			step_output,
			samples,
			radius,
			depth * max_size
			);

		cudaDeviceSynchronize();

		cudaMemcpy(ao.data, gpu_output_data, outBytes, cudaMemcpyDeviceToHost);

		cudaFree(gpu_displacement_data);
		cudaFree(gpu_normal_data);
		cudaFree(gpu_directions_data);
		cudaFree(gpu_output_data);
	}
}