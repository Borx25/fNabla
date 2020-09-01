#pragma once
#pragma warning(disable:4251)

// Windows Header Files
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#include <windows.h>

//MKL
#include "mkl_dfti.h"

//Engine components
#include "fNablaConfig.h"
#include "fNablaEngineCuda.cuh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace fNablaEngine
{

	class __declspec(dllexport) MeshMap {
	protected:
		Config& config;
		double RangeLower = 0.0;
		double RangeUpper = 1.0;
	public:
		cv::Mat Mat;
		int Type = CV_64FC1;
		int ReadFlags = cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH;
		virtual void AllocateMat(const cv::Size shape);
		virtual void Import(const cv::Mat& input, const double factor = 1.0);
		virtual cv::Mat Export(const int depth, const bool postprocess = true, const double factor = 1.0);
		virtual void Normalize();
		virtual cv::Mat Postprocess();

		MeshMap(Config& config_) : config(config_){}


		~MeshMap() {
			Mat.release();
		}
	};

	class __declspec(dllexport) SurfaceMap : public MeshMap {
	public:
		static void ComputeSpectrums(
			std::array<cv::Mat, 3>& spectrums,
			const cv::Size shape,
			const int process_flags,
			Config& config,
			const double scale_factor
		);
		static cv::Mat AllocateSpectrum(cv::Size shape);
		virtual void CalculateSpectrum(cv::Mat& Spectrum);
		virtual void ReconstructFromSpectrum(cv::Mat& Spectrum);

		SurfaceMap(Config& config_) : MeshMap(config_){}
	};

	class __declspec(dllexport) DisplacementMap : public SurfaceMap {
	public:
		DisplacementMap(Config& config_) : SurfaceMap(config_){}
		cv::Mat Postprocess();
	};

	class __declspec(dllexport) NormalMap: public SurfaceMap {
	public:
		NormalMap(Config& config_) : SurfaceMap(config_){
			Type = CV_64FC3;
			RangeLower = -1.0;
			ReadFlags = cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH;
		}

		void Normalize();
		void ReconstructZComponent();
		void CalculateSpectrum(cv::Mat& Spectrum);
		void ReconstructFromSpectrum(cv::Mat& Spectrum);
		cv::Mat Postprocess();
	};

	class __declspec(dllexport) CurvatureMap : public SurfaceMap {
	public:
		CurvatureMap(Config& config_) : SurfaceMap(config_) {
			RangeLower = -1.0;
		}

		void Normalize();
		cv::Mat Postprocess();
	};

	class __declspec(dllexport) AmbientOcclusionMap : public MeshMap {
	public:
		AmbientOcclusionMap(Config& config_) : MeshMap(config_) {}

		void Compute(std::shared_ptr<fNablaEngine::MeshMap> displacement, std::shared_ptr<fNablaEngine::MeshMap> normal);
		cv::Mat Postprocess();
	};

	using MeshMapArray = std::array<std::shared_ptr<fNablaEngine::MeshMap>, fNablaEngine::NUM_OUTPUTS>;

	std::tuple<double, double> GetAlphaBeta(const int CV_Depth, const double lower, const double upper, const bool inverse);

	void __declspec(dllexport) Compute(MeshMapArray& Maps, int process_flags, fNablaEngine::Config& config, double scale_factor);

	void fft2(cv::Mat& input, bool inverse = false);
	void ifft2(cv::Mat& input);
	cv::Mat fftshift(const cv::Mat& input);


#ifdef _DEBUG
	void __declspec(dllexport) image_show(cv::Mat& input, const std::string& name, bool fft = false);
#endif
}