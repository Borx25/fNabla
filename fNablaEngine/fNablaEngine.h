#pragma once
#pragma warning(disable:4251)

// DLL entry
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#include <windows.h>

//Engine components
#include "fNablaConfig.h"
#include "fNablaEngineCuda.cuh"

//Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//MKL
#include "mkl_dfti.h"

namespace fNablaEngine
{

	class __declspec(dllexport) MeshMap {
	protected:
		Configuration& config;
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

		MeshMap(Configuration& config_) : config(config_){}


		~MeshMap() {
			Mat.release();
		}
	};

	class __declspec(dllexport) SurfaceMap : public MeshMap {
	public:
		static void ComputeSpectrums(
			std::array<cv::Mat, 3>& spectrums,
			const cv::Size shape,
			Configuration& configuration,
			Descriptor& descriptor,
			const double scale_factor
		);
		static cv::Mat AllocateSpectrum(cv::Size shape);
		virtual void CalculateSpectrum(cv::Mat& Spectrum);
		virtual void ReconstructFromSpectrum(cv::Mat& Spectrum);

		SurfaceMap(Configuration& config_) : MeshMap(config_){}
	};

	class __declspec(dllexport) DisplacementMap : public SurfaceMap {
	public:
		DisplacementMap(Configuration& config_) : SurfaceMap(config_){}
		cv::Mat Postprocess();
	};

	class __declspec(dllexport) NormalMap: public SurfaceMap {
	public:
		NormalMap(Configuration& config_) : SurfaceMap(config_){
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
		CurvatureMap(Configuration& config_) : SurfaceMap(config_) {
			RangeLower = -1.0;
		}

		void Normalize();
		cv::Mat Postprocess();
	};

	class __declspec(dllexport) AmbientOcclusionMap : public MeshMap {
	public:
		AmbientOcclusionMap(Configuration& config_) : MeshMap(config_) {}
		cv::Mat Postprocess();
	};

	using MeshMapArray = std::array<std::shared_ptr<fNablaEngine::MeshMap>, fNablaEngine::NUM_OUTPUTS>;

	std::tuple<double, double> GetAlphaBeta(const int CV_Depth, const double lower, const double upper, const bool inverse);

	void __declspec(dllexport) ExecuteConversion(MeshMapArray& Maps, Configuration& config, Descriptor& descriptor, double scale_factor = 1.0);

	void fft2(cv::Mat& input, bool inverse = false);
	void ifft2(cv::Mat& input);
	cv::Mat fftshift(const cv::Mat& input);


#ifdef _DEBUG
	void __declspec(dllexport) image_show(cv::Mat& input, const std::string& name, bool fft = false);
#endif
}