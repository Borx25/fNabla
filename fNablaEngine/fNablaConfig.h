#pragma once

#include <string>
#include <regex>
#include <opencv2/core/core.hpp>
#include <functional>

namespace fNablaEngine
{
	enum MapTypes {
		NUM_INPUTS = 3,
		NUM_OUTPUTS = 4,

		DISPLACEMENT = 0,
		NORMAL = 1,
		CURVATURE = 2,
		AO = 3,
	};

	enum ComputeFlags
	{
		INPUT_MASK = ((1 << NUM_INPUTS) - 1) << NUM_OUTPUTS,
		INPUT_DISPLACEMENT = 1 << (DISPLACEMENT + NUM_OUTPUTS),
		INPUT_NORMAL = 1 << (NORMAL + NUM_OUTPUTS),
		INPUT_CURVATURE = 1 << (CURVATURE + NUM_OUTPUTS),

		OUTPUT_MASK = (1 << NUM_OUTPUTS) - 1,
		OUTPUT_DISPLACEMENT = 1 << DISPLACEMENT,
		OUTPUT_NORMAL = 1 << NORMAL,
		OUTPUT_CURVATURE = 1 << CURVATURE,
		OUTPUT_AO = 1 << AO,

		OUTPUT_SURFACEMAPS = OUTPUT_DISPLACEMENT | OUTPUT_NORMAL | OUTPUT_CURVATURE,
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

	static const std::array<std::string, _NUM_FORMATS> Extensions{
		"TIFF",
		"PNG"
	};

	static const std::array<std::array<bool, _NUM_BITDEPTHS>, _NUM_FORMATS> IsDepthSupported{
		{{true, true, true},
		{false, true, true}}
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

			void Set(T new_value) {
				T clamped = std::min(std::max(new_value, m_min), m_max);
				m_val_raw = clamped;
				m_val = (m_transform ? m_transform(clamped) : clamped);
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

		class __declspec(dllexport) Export {
		public:
			Numeric<int> format{ 0, 0, _NUM_FORMATS };
			Numeric<int> bitdepth{ 0, 0, _NUM_BITDEPTHS };
			ValidatedString suffix{ "_default", "[a-zA-Z0-9_-]{2,25}" };

			Export(int format_value, int bitdepth_value, std::string suffix_value);
			void Set_format(int new_value);
			void Set_bitdepth(int new_value);
			void Set_suffix(std::string new_value);

			int Get_format();
			int Get_bitdepth();
			int Get_CVdepth();
			std::string Get_suffix();
			std::string Get_full_suffix();
		};
	}

	struct __declspec(dllexport) Config {
		//Global
		Config_Elements::Numeric<double> integration_window{ 0.000001, 1.0, 1.0, [](double x) -> double { return exp(-16.0 * x); } };
		//Displacement
		Config_Elements::Numeric<int> displacement_colormap{ 0, GRAYSCALE, _NUM_COLORMAPS };
		//Normal
		Config_Elements::Numeric<double> normal_scale{ 0.000001, 0.5, 1.0, [](double x) -> double { return x * 5.0; } };
		Config_Elements::ChannelSign normal_swizzle{ 0, 0 };
		//Curvature
		Config_Elements::Numeric<int> curvature_mode{ 0, CURVATURE_COMPLETE, _NUM_CURVATUREMODES };
		Config_Elements::Numeric<double> curvature_scale{ 0.000001, 0.5, 1.0, [](double x) -> double { return x * 5.0; } };
		//AO
		Config_Elements::Numeric<double> ao_scale{ 0.000001, 0.5, 1.0 };
		Config_Elements::Numeric<int> ao_samples{ 8, 16, 128 };
		Config_Elements::Numeric<double> ao_distance{ 0.000001, 0.35, 1.0};
		Config_Elements::Numeric<double> ao_power{ 0.000001, 0.45, 1.0, [](double x) -> double { return x * 3.0; } };
		//Export
		std::array<Config_Elements::Export, NUM_OUTPUTS>export_settings{ {
			{ TIFF, BIT32, "_displacement" },
			{ PNG, BIT16, "_normal" },
			{ PNG, BIT16, "_curvature" },
			{ PNG, BIT16, "_ambient_occlusion" },
		} };
		//Enabled maps
		std::array<Config_Elements::Boolean, NUM_OUTPUTS>enabled_maps{ {
			{true},
			{true},
			{true},
			{true}
		} };
	};

	struct Viridis {
		Viridis(const Viridis&) = delete;
		Viridis() {}
		static Viridis& Get();
		static cv::Point3d at(double x);

		double R_coeffs[11] = {
			0.2687991305597938,
			0.195695068597545,
			2.069193214801474,
			-32.74409351551069,
			32.00582675786834,
			641.8213302764933,
			-3124.577583332242,
			6561.962859200739,
			-7221.657271442402,
			4071.0458672486193,
			-929.4027186597173,
		};
		double G_coeffs[11] = {
			0.0024819165042698,
			1.6241485718278847,
			-3.525653178812067,
			20.025165047665833,
			-76.0740427442855,
			153.34602434143605,
			-134.14808123604527,
			-27.623014548781104,
			154.33954247241087,
			-117.42885026019304,
			30.368757730257524,
		};
		double B_coeffs[11] = {
			0.3289447267590451,
			1.6020903777881879,
			-3.7756278891864845,
			19.950263860825128,
			-186.39314640723353,
			836.4228957104253,
			-1969.5554553426439,
			2614.45997632588,
			-1948.3737371462446,
			739.1261566689692,
			-103.64168838177879,
		};
	};

	struct Magma {
		Magma(const Magma&) = delete;
		Magma() {}
		static Magma& Get();
		static cv::Point3d at(double x);

		double R_coeffs[11] = {
			0.0004062763460296,
			0.2765580349156502,
			6.066078845033149,
			-15.785757784030428,
			141.79465152894602,
			-1010.5284601954093,
			3470.9812106926493,
			-6369.328232133341,
			6464.36126183672,
			-3433.588054405529,
			746.7387907132164,
		};
		double G_coeffs[11] = {
			0.0069230208847109,
			-0.927625044888057,
			45.25493630314179,
			-529.5616637640426,
			2976.8215782803554,
			-9380.063372551465,
			17758.797776420426,
			-20613.71150768797,
			14347.739661261421,
			-5488.721941663776,
			885.3555318129548,
		};
		double B_coeffs[11] = {
			0.0062637587647372,
			2.640656213391912,
			-32.54058593086639,
			548.1248069266028,
			-3997.301754546941,
			15336.155185166343,
			-34548.07232423682,
			47349.901020990794,
			-38851.35544419425,
			17553.826465839295,
			-3360.636277605164,
		};
	};
}
