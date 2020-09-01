#include "fNablaBatch.h"

int main(int argc, char *argv[]){
    QCoreApplication app(argc, argv);

	QCoreApplication::setApplicationName("fNabla");
	QCoreApplication::setOrganizationName("fNabla");

	app.processEvents();
	fNablaBatch::Task* task = new fNablaBatch::Task(&app);
	QObject::connect(task, SIGNAL(finished()), &app, SLOT(quit()));
	QTimer::singleShot(0, task, SLOT(run()));

    return app.exec();
}


void fNablaBatch::Task::run(){
	QSettings::setDefaultFormat(QSettings::IniFormat);
	QTextStream stream(stdout);

	QCommandLineParser parser;
	parser.setApplicationDescription("fNabla is a tool for conversion between height, normal and curvature maps. Copyright (C) 2020 Borja Franco Garcia");
	parser.addHelpOption();
	parser.addVersionOption();

	parser.addPositionalArgument("Input", QCoreApplication::translate("main", "Input directory."));
	parser.addPositionalArgument("Output", QCoreApplication::translate("main", "Output directory."));

	QCommandLineOption displacementOption(QStringList() << "d" << "displacement", QCoreApplication::translate("main", "Output displacement maps"));
	parser.addOption(displacementOption);
	QCommandLineOption normalOption(QStringList() << "n" << "normal", QCoreApplication::translate("main", "Output normal maps"));
	parser.addOption(normalOption);
	QCommandLineOption curvatureOption(QStringList() << "c" << "curvature", QCoreApplication::translate("main", "Output curvature maps"));
	parser.addOption(curvatureOption);
	QCommandLineOption ambientOcclusionOption(QStringList() << "o" << "occlusion", QCoreApplication::translate("main", "Output curvature maps"));
	parser.addOption(ambientOcclusionOption);


	parser.process(QCoreApplication::arguments());

	const QStringList args = parser.positionalArguments(); //0=source, 1=target
	const QFileInfo inputDir(args[0]);
	const QFileInfo outputDir(args[1]);

	if ((args[0].isEmpty()) || (args[1].isEmpty()) || (!inputDir.exists()) || (!inputDir.isDir()) || (!outputDir.exists()) || (!outputDir.isDir()) || (!outputDir.isWritable())) {
		stream << "[fNabla] [ERROR] Input or output directory do not exist, are not a directory, or output is not writeable. Use parameter -h for more information." << endl;
		emit finished();
		return;
	}

	stream << "[fNabla] [INPUT]: " << inputDir.absoluteFilePath() << endl;
	stream << "[fNabla] [OUTPUT]: " << outputDir.absoluteFilePath() << endl;

	//determine input type based on coincidence with the suffixes of the settings
	bool input_displacement = false;
	bool input_normal = false;
	bool input_curvature = true;

	bool task_displacement = parser.isSet(displacementOption);
	bool task_normal = parser.isSet(normalOption);
	bool task_curvature = parser.isSet(curvatureOption);
	bool task_ao = parser.isSet(ambientOcclusionOption);

	if (!(task_displacement || task_normal || task_curvature || task_ao)) {
		stream << "[fNabla] [ERROR] No conversion selected. Use parameter -h for more information." << endl;
		emit finished();
		return;
	}

	if (task_displacement) stream << "[fNabla] [TASKS]: DISPLACEMENT" << endl;
	if (task_normal) stream << "[fNabla] [TASKS]: NORMAL" << endl;
	if (task_curvature) stream << "[fNabla] [TASKS]: CURVATURE" << endl;
	if (task_ao) stream << "[fNabla] [TASKS]: AMBIENT OCCLUSION" << endl;

	int compute_plan =
		(input_displacement << fNablaEngine::DISPLACEMENT |
		input_normal << fNablaEngine::NORMAL |
		input_curvature << fNablaEngine::CURVATURE) << fNablaEngine::NUM_OUTPUTS |

		task_displacement << fNablaEngine::DISPLACEMENT |
		task_normal << fNablaEngine::NORMAL |
		task_curvature << fNablaEngine::CURVATURE |
		task_ao << fNablaEngine::AO;

	QSettings* app_settings = new QSettings();
	QRegExp* SuffixRegex = new QRegExp("[a-zA-Z0-9_-]{2,25}");

	double Settings_depth = app_settings->value("Global/depth", 0.25).toDouble();
	double Settings_high_pass = app_settings->value("Global/window", 1.0).toDouble();
	double Settings_curvature_sharpness = app_settings->value("Curvature/sharpness", 0.35).toDouble();
	double Settings_ao_samples = app_settings->value("AO/samples", 16.0).toDouble();
	double Settings_ao_distance = app_settings->value("AO/distance", 0.35).toDouble();
	double Settings_ao_power = app_settings->value("AO/power", 0.45).toDouble();
	int swizzle_x = app_settings->value("Normal/swizzle_x", 0).toInt();
	int swizzle_y = app_settings->value("Normal/swizzle_y", 0).toInt();

	int Settings_displacement_format = app_settings->value("Displacement/format", 0).toInt();
	int Settings_displacement_depth = app_settings->value("Displacement/bitdepth", 0).toInt();
	QString Settings_displacement_suffix = app_settings->value("Displacement/suffix", QStringLiteral("_displacement")).toString();
	Settings_displacement_suffix = (SuffixRegex->exactMatch(Settings_displacement_suffix) ? Settings_displacement_suffix : QStringLiteral("_displacement"));

	int Settings_normal_format = app_settings->value("Normal/format", 1).toInt();
	int Settings_normal_depth = app_settings->value("Normal/bitdepth", 1).toInt();
	QString Settings_normal_suffix = app_settings->value("Normal/suffix", QStringLiteral("_normal")).toString();
	Settings_normal_suffix = (SuffixRegex->exactMatch(Settings_normal_suffix) ? Settings_normal_suffix : QStringLiteral("_normal"));

	int Settings_curvature_format = app_settings->value("Curvature/format", 1).toInt();
	int Settings_curvature_depth = app_settings->value("Curvature/bitdepth", 1).toInt();
	QString Settings_curvature_suffix = app_settings->value("Curvature/suffix", QStringLiteral("_curvature")).toString();
	Settings_curvature_suffix = (SuffixRegex->exactMatch(Settings_curvature_suffix) ? Settings_curvature_suffix : QStringLiteral("_curvature"));

	int Settings_ao_format = app_settings->value("AO/format", 1).toInt();
	int Settings_ao_depth = app_settings->value("AO/bitdepth", 1).toInt();
	QString Settings_ao_suffix = app_settings->value("AO/suffix", QStringLiteral("_ambient_occlusion")).toString();
	Settings_ao_suffix = (SuffixRegex->exactMatch(Settings_ao_suffix) ? Settings_ao_suffix : QStringLiteral("_ambient_occlusion"));

	stream << "plan: " << compute_plan << endl;

	const QStringList suffixes = QStringList() << Settings_displacement_suffix << Settings_normal_suffix << Settings_curvature_suffix << Settings_ao_suffix;

	auto local_config = fNablaEngine::Config();

	QDirIterator inputs(inputDir.absoluteFilePath(), QStringList() << "*.png" << "*.tiff" << "*.tif" << "*.pbm", QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories | QDirIterator::FollowSymlinks);
	while (inputs.hasNext()) {
		QFileInfo file(inputs.next());
		stream << "[fNabla][INPUT]: " << file.baseName() << endl;
		int input_type = -1;
		for (int i = 0; i < suffixes.length(); i++)
		{
			if (file.baseName().endsWith(suffixes[i], Qt::CaseSensitive))
			{
				input_type = i;
			}
		}
		stream << "type: " << input_type << endl;
		if (input_type >= 0) //skip if it doesn't contain any of out suffixes
		{
			fNablaEngine::MeshMap* Maps[4] = {
				new fNablaEngine::DisplacementMap(local_config),
				new fNablaEngine::NormalMap(local_config),
				new fNablaEngine::CurvatureMap(local_config),
				new fNablaEngine::AmbientOcclusionMap(local_config),
			};

			cv::Mat input_image = cv::imread(file.absoluteFilePath().toStdString(), Maps[input_type]->ReadFlags);
			const cv::Size shape = input_image.size();

			Maps[input_type]->Import(input_image);

			for (int i = 0; i < 4; i++) {
				if ((i != input_type) && (compute_plan & (1 << i))) {
					stream << "allocate: " << i << endl;
					Maps[i]->Mat = cv::Mat(shape, Maps[i]->Type);
				}
			}

			cv::Mat* spectrums[3] = {
				(compute_plan & fNablaEngine::OUTPUT_DISPLACEMENT ? Maps[fNablaEngine::DISPLACEMENT]->AllocateSpectrum() : nullptr),
				(compute_plan & fNablaEngine::OUTPUT_NORMAL ? Maps[fNablaEngine::NORMAL]->AllocateSpectrum() : nullptr),
				(compute_plan & fNablaEngine::OUTPUT_CURVATURE ? Maps[fNablaEngine::CURVATURE]->AllocateSpectrum() : nullptr),
			};

			Maps[input_type]->Normalize();
			Maps[input_type]->CalculateSpectrum();

			fNablaEngine::ComputeSpectrums(
				spectrums,
				shape,
				compute_plan,
				local_config,
				//Settings_high_pass,
				//Settings_curvature_sharpness,
				1.0
			);

			for (int i = 0; i < 3; i++) {
				if (compute_plan & (1 << i))
				{
					Maps[i]->ReconstructFromSpectrum();
					Maps[i]->Normalize();
				}
			}

			if (compute_plan & fNablaEngine::OUTPUT_AO) {
				dynamic_cast<fNablaEngine::AmbientOcclusionMap*>(Maps[fNablaEngine::AO])->Compute(Maps[fNablaEngine::DISPLACEMENT], Maps[fNablaEngine::NORMAL]);
			}

			QString MapsSaved;
			for (int i = 0; i < 3; i++) {
				if (compute_plan & (1 << i))
				{
					QString output = outputDir.absolutePath() + "/" + file.baseName() + suffixes[i] + "." + "TIFF"; //replace tiff with the format read from setting
					cv::Mat save_img = Maps[i]->Export(int(-2.5 * (double)0 + 5.0)); //0 being the index of export depth
					cv::imwrite(output.toStdString(), save_img);
					MapsSaved.append(output + QString("\n"));
				}
			}

			stream << "[fNabla][SAVED]: " << MapsSaved << endl;
		}
	}

	emit finished();
}