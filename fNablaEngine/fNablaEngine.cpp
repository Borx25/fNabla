#include "fNablaEngine.h"

//----------MESHMAP----------

#define CHECK_MAT if (Mat.empty()) throw std::invalid_argument("Unallocated matrix")
#define CHECK_INPUT(InputMat) if (InputMat.empty()) throw std::invalid_argument("Unallocated matrix")

/// <summary>
/// Allocates the memory required for the input shape on this->Mat. Must be called before any function used the data.
/// </summary>
/// <param name="shape">shape of matrix</param>
void fNablaEngine::TextureMap::AllocateMat(const cv::Size shape) {
	Mat = cv::Mat(shape, Type);
}

/// <summary>
/// Loads a cv::Mat into the Meshmap converting to range and type if needed.
/// </summary>
/// <param name="input">Input data</param>
/// <param name="factor">scale factor</param>
void fNablaEngine::TextureMap::Import(const cv::Mat& input, const double factor){
	CHECK_INPUT(input);

	//extract alpha channel if present
	cv::Mat mask;
	if (input.channels() == 4) {
		mask = cv::Mat(input.rows, input.cols, CVType(input.depth(), 1));
		int mask_fromTo[] = {3, 0};
		cv::mixChannels(&input, 1, &mask, 1, mask_fromTo, 1);

		auto [mask_alpha, mask_beta] = GetAlphaBeta(input.depth(), 0, 1, false);
		mask.convertTo(mask, CVType(Depth, 1), mask_alpha, mask_beta);
	}

	//Handle channels
	std::vector<int> fromTo;
	fromTo.reserve(size_t(NumChannels) * 2);
	for (int i = 0; i < NumChannels; i++) {
		fromTo.push_back(i % input.channels());
		fromTo.push_back(i);
	}
	Mat = cv::Mat(input.rows, input.cols, CVType(input.depth(), NumChannels));
	cv::mixChannels(&input, 1, &Mat, 1, fromTo.data(), NumChannels);

	//Handle data range
	auto [alpha, beta] = GetAlphaBeta(input.depth(), RangeLower, RangeUpper, false);
	Mat.convertTo(Mat, Depth, alpha, beta);

	//apply alpha channel to result
	if (!mask.empty()) {
		if (NumChannels == 1) {
			cv::multiply(Mat, mask, Mat);
		} else {
			Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
				double& m = mask.at<double>(pos);
				p.x *= m;
				p.y *= m;
				p.z *= m;
			});
		}
	}

	//optional resizing
	if (factor != 1.0) {
		cv::resize(Mat, Mat, cv::Size(), factor, factor, cv::INTER_AREA);
	}
}

/// <summary>
/// Exports a copy of the Mat at desired depth with (optionally) the postprocess function applied.
/// </summary>
/// <param name="depth">CV depth</param>
/// <param name="postprocess">Apply postprocess</param>
/// <param name="factor">scale factor</param>
/// <returns>Final cv::Mat</returns>
cv::Mat fNablaEngine::TextureMap::Export(const int depth, const bool postprocess, const double factor) {
	CHECK_MAT;
	auto [alpha, beta] = GetAlphaBeta(depth, RangeLower, RangeUpper, true);
	cv::Mat output = (postprocess ? Postprocess() : Mat.clone());
	output.convertTo(output, output.depth(), alpha, beta);
	//gamma
	//if (output.channels() == 3) {
	//	output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
	//		p.x = pow(p.x, 2.2);
	//		p.y = pow(p.y, 2.2);
	//		p.z = pow(p.z, 2.2);
	//	});
	//} else {
	//	output.forEach<double>([&](double& p, const int* pos) -> void {
	//		p = pow(p, 2.2);
	//	});
	//}
	if (factor != 1.0) {
		cv::resize(output, output, cv::Size(), factor, factor, cv::INTER_AREA);
	}
	output.convertTo(output, depth, 1.0, 0.0);
	return output;
}

/// <summary>
/// Normalize data. Specific behaviour depends on the MeshMap.
/// </summary>
void fNablaEngine::TextureMap::Normalize() {
	CHECK_MAT;
	cv::normalize(Mat, Mat, RangeLower, RangeUpper, cv::NORM_MINMAX);
}

/// <summary>
/// Applies some final modification to the data that is not computationally intensive and requires higher iteration on parameters.
/// </summary>
/// <returns>Copy of Mat with posprocessing applied</returns>
cv::Mat fNablaEngine::TextureMap::Postprocess() {
	CHECK_MAT;
	return Mat.clone();
}


//----------SURFACE MAPS----------

/// <summary>
/// Allocates a matrix of complex doubles to be used in fourier transform of the given shape
/// </summary>
/// <param name="shape"></param>
/// <returns>Matrix of complex doubles</returns>
cv::Mat fNablaEngine::SurfaceMap::AllocateSpectrum(cv::Size shape) {
	return cv::Mat_<std::complex<double>>(shape, std::complex<double>(0.0, 0.0));
}

/// <summary>
/// Forward fourier transform of Mat to spectrum.
/// </summary>
/// <param name="Spectrum">Output spectrum</param>
void fNablaEngine::SurfaceMap::CalculateSpectrum(cv::Mat& Spectrum) {
	CHECK_MAT;
	int fromTo[] = { 0, 0, -1, 1 };
	cv::mixChannels(&Mat, 1, &Spectrum, 1, fromTo, 2);
	fft2(Spectrum);
}

/// <summary>
/// Backwards fourier transform of spectrum to Mat. Note the operation happens in-place and then relevant channels are copied to Mat.
/// </summary>
/// <param name="Spectrum">Input spectrum</param>
void fNablaEngine::SurfaceMap::ReconstructFromSpectrum(cv::Mat& Spectrum) {
	CHECK_INPUT(Spectrum);
	ifft2(Spectrum);
	int fromTo[] = { 0, 0 };
	cv::mixChannels(&Spectrum, 1, &Mat, 1, fromTo, 1);
}

//----------DISPLACEMENT----------

cv::Mat fNablaEngine::DisplacementMap::Postprocess() {
	CHECK_MAT;
	int current_colormap = config.displacement_colormap.Get();
	if (current_colormap == GRAYSCALE) {
		return Mat.clone();
	} else {
		cv::Mat output(Mat.rows, Mat.cols, CV_64FC3);

		switch (current_colormap) {
			case VIRIDIS:
				output.forEach<cv::Point3d>([&](cv::Point3d& display, const int* pos) -> void {
					const double grayscale = Mat.at<double>(pos);
					display = Viridis::at(grayscale);
				});
				break;
			case MAGMA:
				output.forEach<cv::Point3d>([&](cv::Point3d& display, const int* pos) -> void {
					const double grayscale = Mat.at<double>(pos);
					display = Magma::at(grayscale);
				});
				break;
		}
		return output;
	}
}

//----------NORMAL----------

void fNablaEngine::NormalMap::Normalize() {
	CHECK_MAT;
	Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		double norm = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
		if (norm != 0.0) {
			p.x /= norm;
			p.y /= norm;
			p.z /= norm;
		} else {
			p.x = 0.0;
			p.y = 0.0;
			p.z = 0.0;
		}
	});
}

void fNablaEngine::NormalMap::CalculateSpectrum(cv::Mat& Spectrum) {
	CHECK_MAT;
	int fromTo[] = { 1, 0, 2, 1 };
	cv::mixChannels(&Mat, 1, &Spectrum, 1, fromTo, 2);
	cv::multiply(Spectrum, config.normal_swizzle.Get(), Spectrum);
	fft2(Spectrum);
}

void fNablaEngine::NormalMap::ReconstructFromSpectrum(cv::Mat& Spectrum) {
	CHECK_INPUT(Spectrum);
	ifft2(Spectrum);
	cv::multiply(Spectrum, config.normal_swizzle.Get(), Spectrum);
	cv::Mat planes[2];
	cv::split(Spectrum, planes);

	double min, max;
	cv::minMaxLoc(planes[0], &min, &max);
	planes[0] /= std::max(abs(min), abs(max));

	cv::minMaxLoc(planes[1], &min, &max);
	planes[1] /= std::max(abs(min), abs(max));

	int build_normal[] = { 1, 2, 0, 1, -1, 0 }; //XY->BGR
	cv::mixChannels(planes, 2, &Mat, 1, build_normal, 3);

	Mat.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		p.x = 1.0 / sqrt(1.0 + p.y * p.y + p.z * p.z);
		if (p.x != 0.0) {
			p.y *= p.x;
			p.z *= p.x;
		} else {
			p.x = 1.0;
			p.y = 0.0;
			p.z = 0.0;
		}
	});
	Normalize();
}

cv::Mat fNablaEngine::NormalMap::Postprocess() {
	CHECK_MAT;
	cv::Mat output = Mat.clone();
	//derivative
	//output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
	//	p.z = -p.z / p.x;
	//	p.y = -p.y / p.x;
	//	p.x = -1.0;
	//});
	output.forEach<cv::Point3d>([&](cv::Point3d& p, const int* pos) -> void {
		//scale x, y
		p.y = p.y * config.normal_scale.Get();
		p.z = p.z * config.normal_scale.Get();
		//normalize
		p.x = 1.0 / sqrt(1.0 + p.y * p.y + p.z * p.z);
		if (p.x != 0.0) {
			p.y *= p.x;
			p.z *= p.x;
		} else {
			p.x = 1.0;
			p.y = 0.0;
			p.z = 0.0;
		}
	});
	return output;
}

//----------CURVATURE----------

void fNablaEngine::CurvatureMap::Normalize() {
	CHECK_MAT;
	double min, max;
	cv::minMaxLoc(Mat, &min, &max);
	Mat /= std::max(abs(min), max);
}

cv::Mat fNablaEngine::CurvatureMap::Postprocess() {
	CHECK_MAT;
	int current_mode = config.curvature_mode.Get();
	if (current_mode == CURVATURE_SPLIT)
	{
		cv::Mat output(Mat.rows, Mat.cols, CV_64FC3);
		output.forEach<cv::Point3d>([&](cv::Point3d& curv, const int* pos) -> void {
			const double curv_gray = tanh(Mat.at<double>(pos) * config.curvature_scale.Get());
			curv.z = (curv_gray > 0.0 ? curv_gray : 0.0) * 2.0 - 1.0;
			curv.y = (curv_gray < 0.0 ? -curv_gray : 0.0) * 2.0 - 1.0;
			curv.x = -1.0;
		});
		return output;
	}
	else
	{
		cv::Mat output = Mat.clone();
		output.forEach<double>([&](double& curv, const int* pos) -> void {
			curv = tanh(curv * config.curvature_scale.Get());
		});
		switch (current_mode)
		{
		case CURVATURE_CONVEXITY:
			output.forEach<double>([&](double& curv, const int* pos) -> void {
				curv = (curv > 0.0 ? curv : 0.0) * 2.0 - 1.0;
			});
			break;
		case CURVATURE_CONCAVITY:
			output.forEach<double>([&](double& curv, const int* pos) -> void {
				curv = (curv < 0.0 ? -curv : 0.0) * 2.0 - 1.0;
			});
			break;
		}
		return output;
	}
}

//----------AO----------

cv::Mat fNablaEngine::AmbientOcclusionMap::Postprocess() {
	CHECK_MAT;
	cv::Mat output = Mat.clone();
	output.forEach<double>([&](double& occ, const int* pos) -> void {
		occ = 1.0 - pow(1.0 - tanh(occ), config.ao_power.Get());
	});
	return output;
}

//----------METHODS----------
/// <summary>
/// Asyncronously executes the conversion defined by the descriptor on the MeshMaps and tracks its progress
/// </summary>
/// <param name="Maps">Container of maps, unused slots can be null pointers</param>
/// <param name="config">Holds all tweakable parameters used</param>
/// <param name="descriptor">Descriptor of input and outputs</param>
/// <param name="scale_factor">Current scale factor being used for consistency</param>
fNablaEngine::ConversionTask::ConversionTask(TextureSet& Maps, Configuration& configuration, Descriptor& descriptor, double scale_factor) {
	output = std::async(std::launch::async, [&]() {Run(std::ref(Maps), std::ref(configuration), std::ref(descriptor), scale_factor);});
}

/// <summary>
/// Checks if task is done
/// </summary>
bool fNablaEngine::ConversionTask::CheckReady() {
	return (output.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
}

/// <summary>
/// Initializes the progress tracker for a number of milestones
/// </summary>
void fNablaEngine::ConversionTask::StartProgress(int num_milestones) {
	m_num_milestones = num_milestones;
	m_milestone_counter = 0;
	progress = 0.0;
}
/// <summary>
/// Advances to the next milestone
/// </summary>
void fNablaEngine::ConversionTask::NextMilestone(std::string new_status) {
	status = new_status;
	m_milestone_counter += 1;
	progress = double(m_milestone_counter) / double(m_num_milestones);
}

/// <summary>
/// Internal function called by the async wrapper above. Refer to ConversionTask's contructor for parameters.
/// </summary>
void fNablaEngine::ConversionTask::Run(TextureSet& Maps, Configuration& configuration, Descriptor& descriptor, double scale_factor) {
	if ((descriptor.Input >= NUM_INPUTS) || (descriptor.Input < 0) || (descriptor.Output.none())) {
		throw std::invalid_argument("Invalid descriptor");
	}

	const cv::Size shape = Maps[descriptor.Input]->Mat.size();

	Descriptor final_descriptor(descriptor);


	if (final_descriptor.Output[AO]) //handle dependencies, those that need to be computed but not in the original descriptor will be then discarded
	{
		final_descriptor.Output.set(DISPLACEMENT, final_descriptor.Input != DISPLACEMENT);
		final_descriptor.Output.set(NORMAL, final_descriptor.Input != NORMAL);
	}

	StartProgress(((final_descriptor.Output[AO]) ? 6 : 5));

	NextMilestone("Allocating matrices...");

	Maps[final_descriptor.Input]->Normalize();

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		if ((final_descriptor.Input != i) && (final_descriptor.Output[i])) {
			Maps[i]->AllocateMat(shape);
		}
	}

	std::array<cv::Mat, 3> spectrums = {
		(((final_descriptor.Output[DISPLACEMENT]) || (final_descriptor.Input == DISPLACEMENT)) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
		(((final_descriptor.Output[NORMAL]) || (final_descriptor.Input == NORMAL)) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
		(((final_descriptor.Output[CURVATURE]) || (final_descriptor.Input == CURVATURE)) ? SurfaceMap::AllocateSpectrum(shape) : cv::Mat()),
	};

	NextMilestone("Obtaining frequency domain...");

	dynamic_cast<SurfaceMap*>(Maps[final_descriptor.Input].get())->CalculateSpectrum(spectrums[final_descriptor.Input]);

	NextMilestone("Computing spectrums...");

	const double aspect_ratio = double(shape.height) / double(shape.width);
	const double inv_aspect_ratio = double(shape.width) / double(shape.height);
	const double effective_scale_factor = _4_PI_PI / (scale_factor * scale_factor);
	const double effective_high_pass = 1.0 / (configuration.integration_window.Get() * effective_scale_factor);

	const cv::Point2d sigma_integration(effective_high_pass * inv_aspect_ratio, effective_high_pass * aspect_ratio);

	spectrums[final_descriptor.Input].forEach<std::complex<double>>([&](std::complex<double>& input, const int* pos) {
	//std::for_each(spectrums[final_descriptor.Input].begin<std::complex<double>>(), spectrums[final_descriptor.Input].end<std::complex<double>>(), [](std::complex<double>& input) {
		const double x = (double)pos[1] / (double)shape.width;
		const double y = (double)pos[0] / (double)shape.height;
		//plot 2 * pi * (x - floor(2x)) + i * 2 * pi * (y - floor(2y)), x = 0..1, y = 0..1
		const double ramp_x = _2_PI * (x - floor(2 * x));
		const double ramp_y = _2_PI * (y - floor(2 * y));
		std::complex<double> ramp(
			ramp_x,
			ramp_y
		);

		//plot 8.0 * cos^2(0.5 * 2*pi*(y-floor(2y))) * sin(2*pi*(x-floor(2x))) + i * 8.0 * cos^2(0.5 * 2*pi*(x-floor(2x))) * sin(2*pi*(y-floor(2y))), x=0..1, y=0..1
		//4.0 * (1 + cos(ramp_y)) * sin(ramp_x) + i * 4.0 * (1 + cos(ramp_y)) * sin(ramp_x)
		std::complex<double> smooth_operator(
			4.0 * (1.0 + cos(ramp_y)) * sin(ramp_x),
			4.0 * (1.0 + cos(ramp_x)) * sin(ramp_y)
		);

		const double integration_window = 1.0 - exp(-(ramp_x * ramp_x * sigma_integration.x + ramp_y * ramp_y * sigma_integration.y));
		const double sq_freq = norm(ramp);

		if (sq_freq != 0.0) {
			if (final_descriptor.Output[DISPLACEMENT]) {
				std::complex<double>& h = spectrums[DISPLACEMENT].at<std::complex<double>>(pos);
				if (final_descriptor.Input == NORMAL) {
					h = input / ramp;
				} else if (final_descriptor.Input == CURVATURE) {
					h = input / sq_freq;
				}
				h *= integration_window;
			}

			if (final_descriptor.Output[NORMAL]) {
				std::complex<double>& n = spectrums[NORMAL].at<std::complex<double>>(pos);
				if (final_descriptor.Input == DISPLACEMENT) {
					n = input * smooth_operator;
				} else if (final_descriptor.Input == CURVATURE) {
					n = (input / conj(ramp)) * integration_window;
				}
			}

			if (final_descriptor.Output[CURVATURE]) {
				std::complex<double>& c = spectrums[CURVATURE].at<std::complex<double>>(pos);
				if (final_descriptor.Input == DISPLACEMENT) {
					c = input * norm(smooth_operator);
				} else if (final_descriptor.Input == NORMAL) {
					c = input * conj(smooth_operator);
				}
			}
		}
	});

	NextMilestone("Reconstructing...");

	for (int i = 0; i < 3; i++) {
		if ((final_descriptor.Input != i) && (final_descriptor.Output[i])) {
			dynamic_cast<SurfaceMap*>(Maps[i].get())->ReconstructFromSpectrum(spectrums[i]);
			Maps[i]->Normalize();
		}
	}
	if (final_descriptor.Output[AO]) {

		NextMilestone("Preparing Ambient Occlusion...");

		fNablaEngineCuda::ComputeAOCuda(
			Maps[DISPLACEMENT]->Mat,
			Maps[NORMAL]->Mat,
			Maps[AO]->Mat,
			configuration.ao_samples.Get(),
			configuration.ao_distance.Get(),
			configuration.ao_scale.Get(),
			status
		);
	}

	NextMilestone("Done!");
}

bool fNablaEngine::CheckGPUCompute() {
	try {
		fNablaEngineCuda::CheckCUDACapable();
		return true;
	} catch (fNablaEngineCuda::NoCUDAGPU&) {
		return false;
	} catch (...) {
		return false;
	}
}

int fNablaEngine::CVType(const int CV_Depth, const unsigned int num_channels) {
	return CV_Depth + (((num_channels)-1) << CV_CN_SHIFT); //following how it's currently done in the implementation
}
/// <summary>
/// Obtain Alpha (scale factor) and Beta (offset) values for the conversion from an arbitrary CV type to the floating point range used internally by the MeshMap
/// </summary>
/// <param name="CV_Depth">integer describing the depth of the image</param>
/// <param name="out_low">lower bound of output range</param>
/// <param name="out_up">upper bound of output range</param>
/// <param name="inverse"></param>
/// <returns>tuple(double alpha, double beta)</returns>
std::tuple<double, double> fNablaEngine::GetAlphaBeta(const int CV_Depth, const double out_low, const double out_up, const bool inverse) {
	double in_low, in_up;
	switch (CV_Depth) {
		case CV_8U:
			in_low = 0.0; in_up = 255.0; break;
		case CV_8S:
			in_low = -128.0; in_up = 127.0; break;
		case CV_16U:
			in_low = 0.0; in_up = 65535.0; break;
		case CV_16S:
			in_low = -32768.0; in_up = 32767.0; break;
		case CV_32S:
			in_low = -2147483648.0; in_up = 2147483647.0; break;
		case CV_32F:
			in_low = 0.0; in_up = 1.0; break;
		case CV_64F:
			in_low = 0.0; in_up = 1.0; break;
		default:
			in_low = 0.0; in_up = 1.0; break;
	}
	const double input_range = in_up - in_low;
	const double output_range = out_up - out_low;
	if (inverse) {
		return {input_range / output_range, (in_low * out_up - in_up * out_low) / output_range};
	} else {
		return {output_range / input_range, out_up - in_up * (output_range / input_range)};
	}
}

/// <summary>
/// Performs in-place forward or backwards fourier transform of input
/// </summary>
/// <param name="input">complex-valued (2 channels) matrix</param>
/// <param name="inverse">forward or backwards</param>
void fNablaEngine::fft2(cv::Mat& input, bool inverse) {
#ifndef USE_MKL
	if (inverse) {
		cv::dft(input, input, cv::DFT_COMPLEX_INPUT || cv::DFT_COMPLEX_OUTPUT || cv::DFT_INVERSE || cv::DFT_SCALE);
	} else {
		cv::dft(input, input, cv::DFT_COMPLEX_INPUT || cv::DFT_COMPLEX_OUTPUT);
	}
#endif // USE_MKL
#ifdef USE_MKL
	DFTI_DESCRIPTOR_HANDLE descriptor = NULL;
	int status;
	int dim_sizes[2] = {input.rows, input.cols};

	//status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim_sizes);
	status = DftiCreateDescriptor_d_md(&descriptor, DFTI_COMPLEX, 2, dim_sizes); //to reduce custom DLL size we are calling directly the double precision function
	if (inverse) {
		status = DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (dim_sizes[0] * dim_sizes[1]));
		status = DftiCommitDescriptor(descriptor);
		status = DftiComputeBackward(descriptor, input.data);
	} else {
		status = DftiCommitDescriptor(descriptor);
		status = DftiComputeForward(descriptor, input.data);
	}
	status = DftiFreeDescriptor(&descriptor);
#endif // USE_MKL
}
/// <summary>
/// Convenience function for inverse fourier transform. See fNablaEngine::fft2.
/// </summary>
/// <param name="input">complex-valued (2 channels) matrix</param>
void fNablaEngine::ifft2(cv::Mat& input) {
	fft2(input, true);
}

/// <summary>
/// Shifts cuadrants of fourier transform
/// </summary>
/// <param name="input">input fourier transform</param>
/// <returns>shifted fourier transform</returns>
cv::Mat fNablaEngine::fftshift(const cv::Mat& input) {
	cv::Mat output = input(cv::Rect(0, 0, input.cols & ~2, input.rows & ~2)); //round down to even integer

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = output.cols / 2;
	int cy = output.rows / 2;
	cv::Mat q0(output, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(output, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(output, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(output, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	return output;
}

fNablaEngine::Viridis& fNablaEngine::Viridis::Get() {
	static Viridis instance;
	return instance;
}

cv::Point3d fNablaEngine::Viridis::at(double x) {
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	Viridis& ref = Viridis::Get();
	for (int i = 10; i >= 0; i--) {
		double power = pow(x, i);
		r += ref.R_coeffs[i] * power;
		g += ref.G_coeffs[i] * power;
		b += ref.B_coeffs[i] * power;
	}
	return cv::Point3d(b, g, r);
}

fNablaEngine::Magma& fNablaEngine::Magma::Get() {
	static Magma instance;
	return instance;
}

cv::Point3d fNablaEngine::Magma::at(double x) {
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	Magma& ref = Magma::Get();
	for (int i = 10; i >= 0; i--) {
		double power = pow(x, i);
		r += ref.R_coeffs[i] * power;
		g += ref.G_coeffs[i] * power;
		b += ref.B_coeffs[i] * power;
	}
	return cv::Point3d(b, g, r);
}