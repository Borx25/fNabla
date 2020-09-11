#include "fNablaBatch.h"

int main(int argc, char *argv[]){
    QCoreApplication app(argc, argv);

	QCoreApplication::setApplicationName("fNabla");
	QCoreApplication::setOrganizationName("fNabla");

	app.processEvents();
	fNablaBatch::Task task(&app);
	QObject::connect(&task, SIGNAL(finished()), &app, SLOT(quit()));
	QTimer::singleShot(0, &task, SLOT(run()));

    return app.exec();
}


void fNablaBatch::Task::run() {

	QTextStream stream(stdout);

	QCommandLineParser parser;
	parser.setApplicationDescription(QStringLiteral("fNabla is a tool for conversion between height, normal and curvature maps. Copyright (C) 2020 Borja Franco Garcia"));
	parser.addHelpOption();
	parser.addVersionOption();

	parser.addPositionalArgument(QStringLiteral("Input"), QCoreApplication::translate("main", "Input directory."));
	parser.addPositionalArgument(QStringLiteral("Output"), QCoreApplication::translate("main", "Output directory."));
	parser.addPositionalArgument(QStringLiteral("Settings"), QCoreApplication::translate("main", "Location of ini file with settings, pass empty string to use defaults"));

	QCommandLineOption displacementOption(QStringList() << QStringLiteral("d") << QStringLiteral("displacement"), QCoreApplication::translate("main", "Output displacement maps"));
	parser.addOption(displacementOption);

	QCommandLineOption normalOption(QStringList() << QStringLiteral("n") << QStringLiteral("normal"), QCoreApplication::translate("main", "Output normal maps"));
	parser.addOption(normalOption);

	QCommandLineOption curvatureOption(QStringList() << QStringLiteral("c") << QStringLiteral("curvature"), QCoreApplication::translate("main", "Output curvature maps"));
	parser.addOption(curvatureOption);

	QCommandLineOption ambientOcclusionOption(QStringList() << QStringLiteral("o") << QStringLiteral("occlusion"), QCoreApplication::translate("main", "Output curvature maps"));
	parser.addOption(ambientOcclusionOption);

	//setup configuration overrides

	parser.process(QCoreApplication::arguments());

	const QStringList args = parser.positionalArguments(); //0=source, 1=target, 2=settings

	if ((args.length() < 2) || (args[0].isEmpty()) || (args[1].isEmpty())) {
		stream << QStringLiteral("[fNabla] [ERROR] Not enough positional parameters. Use parameter -h for more information.") << endl;
		emit finished();
		return;
	}

	const QFileInfo inputDir(args[0]);
	const QFileInfo outputDir(args[1]);

	if ((!inputDir.exists()) || 
		(!inputDir.isDir()) || 
		(!outputDir.exists()) || 
		(!outputDir.isDir()) || 
		(!outputDir.isWritable())
		) {
		stream << QStringLiteral("[fNabla] [ERROR] Input or output directory do not exist, are not a directory, or output is not writeable. Use parameter -h for more information.") << endl;
		emit finished();
		return;
	} else {
		stream << QStringLiteral("[fNabla] [INPUT DIR]: ") << inputDir.absoluteFilePath() << endl;
		stream << QStringLiteral("[fNabla] [OUTPUT DIR]: ") << outputDir.absoluteFilePath() << endl;
	}

	Configuration configuration;
	Descriptor global_descriptor;

	global_descriptor.Output.set(DISPLACEMENT, parser.isSet(displacementOption));
	global_descriptor.Output.set(NORMAL, parser.isSet(normalOption));
	global_descriptor.Output.set(CURVATURE, parser.isSet(curvatureOption));
	global_descriptor.Output.set(AO, parser.isSet(ambientOcclusionOption));

	if (!((global_descriptor.Output[DISPLACEMENT]) || 
		  (global_descriptor.Output[NORMAL]) || 
		  (global_descriptor.Output[CURVATURE]) || 
		  (global_descriptor.Output[AO])
		)) {
		stream << QStringLiteral("[fNabla] [ERROR] No conversion selected. Use parameter -h for more information.") << endl;
		emit finished();
		return;
	} else {
		stream << QStringLiteral("[fNabla] [TASKS]: ");
		for (unsigned int i = 0; i < NUM_OUTPUTS; i++) {
			stream << (global_descriptor.Output[i] ? MapNames[i] + QStringLiteral(" ") : NULL);
		}
		stream << endl;
	}

	if ((args.length() >= 3) && !(args[2].isEmpty())) {
		const QFileInfo settingsIni(args[2]);
		if ((settingsIni.exists()) && (settingsIni.isFile()) && (settingsIni.isReadable()) && (settingsIni.suffix() == QStringLiteral("ini"))) {
			stream << QStringLiteral("[fNabla] [SETTINGS]: ") << settingsIni.absoluteFilePath() << endl;
			QSettings overrides(settingsIni.absoluteFilePath(), QSettings::IniFormat, this);

			configuration.integration_window.Set(overrides.value("Global/window", configuration.integration_window.Get_raw()).toDouble());

			if ((global_descriptor.Output[NORMAL]) || (global_descriptor.Input == NORMAL)) {
				configuration.normal_scale.Set(overrides.value("Normal/scale", configuration.normal_scale.Get_raw()).toDouble());
				configuration.normal_swizzle.Set(
					overrides.value("Normal/swizzle_x", configuration.normal_swizzle.Get_x()).toInt(),
					overrides.value("Normal/swizzle_y", configuration.normal_swizzle.Get_y()).toInt()
				);
			}

			if (global_descriptor.Output[CURVATURE]) {
				configuration.curvature_scale.Set(overrides.value("Curvature/scale", configuration.curvature_scale.Get_raw()).toDouble());
				configuration.curvature_mode.Set(overrides.value("Curvature/mode", configuration.curvature_mode.Get_raw()).toInt());
			}

			if (global_descriptor.Output[AO]) {
				configuration.ao_scale.Set(overrides.value("AO/scale", configuration.ao_scale.Get_raw()).toDouble());
				configuration.ao_samples.Set(overrides.value("AO/samples", configuration.ao_samples.Get_raw()).toDouble());
				configuration.ao_distance.Set(overrides.value("AO/distance", configuration.ao_distance.Get_raw()).toDouble());
				configuration.ao_power.Set(overrides.value("AO/power", configuration.ao_power.Get_raw()).toDouble());
			}

			for (int i = 0; i < NUM_OUTPUTS; i++) {
				if ((global_descriptor.Output[i]) || (global_descriptor.Input == i)) {
					configuration.export_settings[i].Set_format(overrides.value(SettingCategory[i] + QStringLiteral("/format"), configuration.export_settings[i].Get_format()).toInt());
					configuration.export_settings[i].Set_bitdepth(overrides.value(SettingCategory[i] + QStringLiteral("/bitdepth"), configuration.export_settings[i].Get_bitdepth()).toInt());
					configuration.export_settings[i].Set_suffix(overrides.value(SettingCategory[i] + QStringLiteral("/suffix"), QString::fromStdString(configuration.export_settings[i].Get_suffix())).toString().toStdString());
				}
			}
		} else {
			stream << QStringLiteral("[fNabla] [ERROR] Invalid settings file passed. Use parameter -h for more information.") << endl;
			emit finished();
			return;
		}
	}

	QDirIterator inputs(inputDir.absoluteFilePath(), SupportedFormats, QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories | QDirIterator::FollowSymlinks);
	while (inputs.hasNext()) {
		QFileInfo file(inputs.next());
		QString TextureSet;
		Descriptor descriptor(global_descriptor);
		descriptor.Input = -1;
		for (int i = 0; i < NUM_OUTPUTS; i++) {
			QString suffix = QString::fromStdString(configuration.export_settings[i].Get_suffix());
			if (file.baseName().endsWith(suffix, Qt::CaseSensitive)) {
				TextureSet = outputDir.absoluteFilePath() + QStringLiteral("/") + file.baseName().remove(suffix);
				descriptor.Input = i;
			}
		}
		if ((descriptor.Input >= 0) && (descriptor.Input < NUM_OUTPUTS)) //skip if it doesn't contain any of our suffixes, i.e. we can't infer input type
		{
			stream << QStringLiteral("[fNabla][INPUT]: ") << TextureSet << QStringLiteral(" ") << MapNames[descriptor.Input] << endl;

			MeshMapArray Maps = {{
				std::make_shared<DisplacementMap>(configuration),
				std::make_shared<NormalMap>(configuration),
				std::make_shared<CurvatureMap>(configuration),
				std::make_shared<AmbientOcclusionMap>(configuration),
			}};

			cv::Mat input_image = cv::imread(file.absoluteFilePath().toStdString(), Maps[descriptor.Input]->ReadFlags);
			const cv::Size shape = input_image.size();

			Maps[descriptor.Input]->Import(input_image);

			ExecuteConversion(Maps, configuration, descriptor);

			QString MapsSaved;
			for (int i = 0; i < NUM_OUTPUTS; i++) {
				if (descriptor.Output[i]) {
					QString output = TextureSet + QString::fromStdString(configuration.export_settings[i].Get_full_suffix());
					cv::Mat save_img = Maps[i]->Export(configuration.export_settings[i].Get_CVdepth(), i != 0);
					cv::imwrite(output.toStdString(), save_img);
					MapsSaved.append(output + QStringLiteral("\n"));
				}
			}

			stream << QStringLiteral("[fNabla][SAVED]: ") << endl << MapsSaved << endl;
		}
	}

	emit finished();
}