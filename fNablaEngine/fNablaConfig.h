#pragma once

#include <string>
#include <regex>
#include <opencv2/core/core.hpp>
#include <functional>
#include <bitset>

namespace fNablaEngine
{
	enum MapTypes {
		DISPLACEMENT,
		NORMAL,
		CURVATURE,
		AO,
		NUM_OUTPUTS,
		NUM_INPUTS = 3,
	};

	enum Colormap_types {
		GRAYSCALE,
		VIRIDIS,
		MAGMA,
		_NUM_COLORMAPS,
	};

	enum CurvatureModes {
		CURVATURE_COMPLETE,
		CURVATURE_SPLIT,
		CURVATURE_CONVEXITY,
		CURVATURE_CONCAVITY,
		_NUM_CURVATUREMODES
	};

	enum BitDepths {
		BIT32,
		BIT16,
		BIT8,
		_NUM_BITDEPTHS
	};

	enum Formats {
		TIFF, //8,16,32 bitdepth
		PNG, //8,16 bitdepth
		_NUM_FORMATS
	};



	namespace Config_Elements {

		class __declspec(dllexport) Boolean {
		public:
			bool m_val;

			Boolean(bool val);
			void Set(bool val);
			bool Get();

		};

		template <typename T>
		class __declspec(dllexport) Numeric {
		public:
			T m_val_raw;
			T m_min;
			T m_val;
			T m_max;
			std::function<T(T)> m_transform;

			Numeric(T min, T value, T max, std::function<T(T)> transform = 0) {
				m_min = min;
				m_max = max;
				m_transform = transform;
				Set(value);
			}

			T Set(T new_value) {
				T clamped = std::min(std::max(new_value, m_min), m_max);
				m_val_raw = clamped;
				m_val = (m_transform ? m_transform(clamped) : clamped);
				return m_val;
			}

			T Get() {
				return m_val;
			}

			T Get_raw() {
				return m_val_raw;
			}
		};

		class __declspec(dllexport) ValidatedString {
		public:
			std::string m_val;
			std::regex m_validate_regex;

			ValidatedString(std::string value, std::string regex);

			void Set(std::string new_value);
			std::string Get();
		};

		class __declspec(dllexport) ChannelSign {
		public:
			bool m_x;
			bool m_y;
			cv::Scalar m_val;

			ChannelSign(bool x, bool y);
			void Set(bool x, bool y);
			cv::Scalar Get();
			bool Get_x();
			bool Get_y();
		};

		static const std::regex suffix_regex("[a-zA-Z0-9_-]{2,25}");

		class __declspec(dllexport) ExportSettings {
		public:
			int m_format;
			int m_bitdepth;
			std::string m_suffix;

			ExportSettings(int format_value, int bitdepth_value, std::string suffix_value);
			void Set_format(int new_value);
			void Set_bitdepth(int new_value);
			void Set_suffix(std::string new_value);

			int Get_format();
			int Get_bitdepth();
			int Get_CVdepth();
			std::string Get_suffix();
			std::string Get_full_suffix();

			///Format-depth compatibility handling
			bool CheckCompatibility(bool FormatPreference = true);

		};
	}

	/// Describes an operation of Meshmaps
	struct __declspec(dllexport) Descriptor {
		/// Index corresponding to the input MeshMap. Refer to MapTypes enum.
		unsigned int Input = -1;
		/// Bitset describing the chosen outputs. Refer to MapTypes enum for the order.
		std::bitset<NUM_OUTPUTS> Output;
	};

	struct __declspec(dllexport) Configuration {
		//Global
		Config_Elements::Numeric<double> integration_window{ 0.000001, 1.0, 1.0, [](double x) -> double { return exp(-16.0 * x); } };
		//Displacement
		Config_Elements::Numeric<int> displacement_colormap{ 0, GRAYSCALE, _NUM_COLORMAPS };
		//Normal
		Config_Elements::Numeric<double> normal_scale{ 0.000001, 0.135, 1.0, [](double x) -> double { return exp(5.0 * x); } };
		Config_Elements::ChannelSign normal_swizzle{false, false};
		//Curvature
		Config_Elements::Numeric<int> curvature_mode{ 0, CURVATURE_COMPLETE, _NUM_CURVATUREMODES };
		Config_Elements::Numeric<double> curvature_scale{ 0.000001, 0.2, 1.0, [](double x) -> double { return exp(5.0 * x); } };
		//AO
		Config_Elements::Numeric<double> ao_scale{ 0.000001, 0.25, 1.0 };
		Config_Elements::Numeric<int> ao_samples{ 8, 16, 128 };
		Config_Elements::Numeric<double> ao_distance{ 0.000001, 0.3, 1.0};
		Config_Elements::Numeric<double> ao_power{ 0.000001, 0.16, 1.0, [](double x) -> double { return x * 3.0; } };
		//Export
		std::array<Config_Elements::ExportSettings, NUM_OUTPUTS>export_settings{ {
			{ TIFF, BIT32, std::string("_displacement") },
			{ PNG, BIT16, std::string("_normal") },
			{ PNG, BIT16, std::string("_curvature") },
			{ PNG, BIT16, std::string("_ambient_occlusion") },
		} };
		//Enabled maps
		std::array<Config_Elements::Boolean, NUM_OUTPUTS>enabled_maps{ {
			{true},
			{true},
			{true},
			{true}
		} };
	};
}
