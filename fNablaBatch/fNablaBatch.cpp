#include "fNablaBatch.h"

int main(int argc, char *argv[]){
    QCoreApplication app(argc, argv);

	QCoreApplication::setApplicationName("fNabla");
	QCoreApplication::setOrganizationName("fNabla");

	app.processEvents();
	fNablaBatch::Task task(&app);
	QObject::connect(&task, SIGNAL(Finished()), &app, SLOT(quit()));
	QTimer::singleShot(0, &task, SLOT(Run()));

    return app.exec();
}

void fNablaBatch::Task::Run() {

	static const QStringList SettingCategory{
		QStringLiteral("Displacement"),
		QStringLiteral("Normal"),
		QStringLiteral("Curvature"),
		QStringLiteral("AO")
	};

	static const QStringList MapNames{
		QStringLiteral("[DISPLACEMENT]"),
		QStringLiteral("[NORMAL]"),
		QStringLiteral("[CURVATURE]"),
		QStringLiteral("[AMBIENT OCCLUSION]")
	};

	static const QStringList SupportedFormats{
		QStringLiteral("*.png"),
		QStringLiteral("*.tiff"),
		QStringLiteral("*.tif"),
		QStringLiteral("*.pbm")
	};

	QTextStream stream(stdout);

///ABORT MACRO--------
#define ABORT(ERROR_MESSAGE)																														\
stream << QStringLiteral("[fNabla][ERROR] ") << QStringLiteral(ERROR_MESSAGE) << QStringLiteral(" Use parameter -h for more information.") << endl;	\
emit Finished();																																	\
return
///------------------

	QCommandLineParser parser;
	parser.setApplicationDescription(QStringLiteral(
"\nfNabla is a tool for conversion between height, normal, curvature and ambient occlusion maps. Copyright (C) 2020 Borja Franco Garcia \n\n\
> Computing ambient occlusion using the default configuration: \n\
\t \"[Installation folder]\\fNablaBatch.exe\" -o \"[pipeline path]\\in\" \"[pipeline path]\\out\" \n\n\
> Computing all maps using a custom configuration file: \n\
\t \"[Installation folder]\\fNablaBatch.exe\" -dnco \"[pipeline path]\\in\" \"[pipeline path]\\out\" \"[pipeline path]\\my_config_file.ini\" \n\n\
Generating a default configuration file for customization: \n\
\t \"[Installation folder]\\fNablaBatch.exe\" -t \"[pipeline path]\\my_config_file.ini\" \n\n\
Please note ambient occlusion calculation requires a CUDA-capable GPU"
	));
	parser.addHelpOption();
	parser.addVersionOption();

	parser.addPositionalArgument(QStringLiteral("Input"), QCoreApplication::translate("main", "Input directory with files using the suffixes determined by the configuration."));
	parser.addPositionalArgument(QStringLiteral("Output"), QCoreApplication::translate("main", "Output directory."));
	parser.addPositionalArgument(QStringLiteral("Settings"), QCoreApplication::translate("main", "Location of ini file with settings. If missing default values will be used, use option -t to generate a template."));

	QCommandLineOption templateOption(QStringList() << QStringLiteral("t") << QStringLiteral("template"), QCoreApplication::translate("main", "Generate a template configuration file. Must be followed by path in which to write it with extension .ini"));
	parser.addOption(templateOption);

	QCommandLineOption displacementOption(QStringList() << QStringLiteral("d") << QStringLiteral("displacement"), QCoreApplication::translate("main", "Output displacement maps"));
	parser.addOption(displacementOption);

	QCommandLineOption normalOption(QStringList() << QStringLiteral("n") << QStringLiteral("normal"), QCoreApplication::translate("main", "Output normal maps"));
	parser.addOption(normalOption);

	QCommandLineOption curvatureOption(QStringList() << QStringLiteral("c") << QStringLiteral("curvature"), QCoreApplication::translate("main", "Output curvature maps"));
	parser.addOption(curvatureOption);

	QCommandLineOption ambientOcclusionOption(QStringList() << QStringLiteral("o") << QStringLiteral("occlusion"), QCoreApplication::translate("main", "Output curvature maps"));
	parser.addOption(ambientOcclusionOption);

	//Parse parameters

	parser.process(QCoreApplication::arguments());

	const QStringList args = parser.positionalArguments(); //0=source, 1=target, 2=settings

	Configuration configuration;

	if (parser.isSet(templateOption)) {
		if ((args.length() == 1) && !(args[0].isEmpty())) {
			const QFileInfo templateIni(args[0]);
			if (templateIni.suffix() == QStringLiteral("ini")) {
				QSettings overrides(templateIni.absoluteFilePath(), QSettings::IniFormat, this);

				overrides.beginGroup(QStringLiteral("Global"));
				overrides.setValue(QStringLiteral("window"), configuration.integration_window.Get_raw());
				overrides.endGroup();

				overrides.beginGroup(SettingCategory[NORMAL]);
				overrides.setValue(QStringLiteral("scale"), configuration.normal_scale.Get_raw());
				overrides.setValue(QStringLiteral("swizzle_x"), configuration.normal_swizzle.Get_x());
				overrides.setValue(QStringLiteral("swizzle_y"), configuration.normal_swizzle.Get_y());
				overrides.endGroup();

				overrides.beginGroup(SettingCategory[CURVATURE]);
				overrides.setValue(QStringLiteral("scale"), configuration.curvature_scale.Get_raw());
				overrides.setValue(QStringLiteral("mode"), configuration.curvature_mode.Get_raw());
				overrides.endGroup();

				overrides.beginGroup(SettingCategory[AO]);
				overrides.setValue(QStringLiteral("scale"), configuration.ao_scale.Get_raw());
				overrides.setValue(QStringLiteral("samples"), configuration.ao_samples.Get_raw());
				overrides.setValue(QStringLiteral("distance"), configuration.ao_distance.Get_raw());
				overrides.setValue(QStringLiteral("power"), configuration.ao_power.Get_raw());
				overrides.endGroup();

				for (int i = 0; i < NUM_OUTPUTS; i++) {
					overrides.beginGroup(SettingCategory[i]);
					overrides.setValue(QStringLiteral("format"), configuration.export_settings[i].Get_format());
					overrides.setValue(QStringLiteral("bitdepth"), configuration.export_settings[i].Get_bitdepth());
					overrides.setValue(QStringLiteral("suffix"), QString::fromStdString(configuration.export_settings[i].Get_suffix()));
					overrides.endGroup();
				}
				stream << QStringLiteral("[fNabla] Saved template configuration file: ") << templateIni.absoluteFilePath() << endl;
				emit Finished();
				return;
			}
		}
		ABORT("Invalid arguments with parameter - t.");
	}

	if ((args.length() < 2) || (args[0].isEmpty()) || (args[1].isEmpty())) {
		ABORT("Not enough positional parameters.");
	}

	const QFileInfo inputDir(args[0]);
	const QFileInfo outputDir(args[1]);

	if ((!inputDir.exists()) || 
		(!inputDir.isDir()) || 
		(!outputDir.exists()) || 
		(!outputDir.isDir()) || 
		(!outputDir.isWritable())
		) {
		ABORT("Input or output directory do not exist, are not a directory, or output is not writeable.");
	} else {
		stream << QStringLiteral("[fNabla] [INPUT DIR]: ") << inputDir.absoluteFilePath() << endl;
		stream << QStringLiteral("[fNabla] [OUTPUT DIR]: ") << outputDir.absoluteFilePath() << endl;
	}

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
		ABORT("No conversion selected.");
	} else {
		stream << QStringLiteral("[fNabla] [TASKS]: ");
		for (unsigned int i = 0; i < NUM_OUTPUTS; i++) {
			stream << (global_descriptor.Output[i] ? MapNames[i] + QStringLiteral(" ") : NULL);
		}
		stream << endl;
	}

	//Load setting overrides

	if ((args.length() == 3) && !(args[2].isEmpty())) {
		const QFileInfo settingsIni(args[2]);
		if ((settingsIni.exists()) && (settingsIni.isFile()) && (settingsIni.isReadable()) && (settingsIni.suffix() == QStringLiteral("ini"))) {
			stream << QStringLiteral("[fNabla] [SETTINGS]: ") << settingsIni.absoluteFilePath() << endl;
			QSettings overrides(settingsIni.absoluteFilePath(), QSettings::IniFormat, this);

			configuration.integration_window.Set(overrides.value(QStringLiteral("Global/window"), configuration.integration_window.Get_raw()).toDouble());

			overrides.beginGroup(SettingCategory[NORMAL]);
			configuration.normal_swizzle.Set( //could be used by inputs, always load
				overrides.value(QStringLiteral("swizzle_x"), configuration.normal_swizzle.Get_x()).toInt(),
				overrides.value(QStringLiteral("swizzle_y"), configuration.normal_swizzle.Get_y()).toInt()
			);

			if (global_descriptor.Output[NORMAL]) {
				configuration.normal_scale.Set(overrides.value(QStringLiteral("scale"), configuration.normal_scale.Get_raw()).toDouble());
			}
			overrides.endGroup();

			if (global_descriptor.Output[CURVATURE]) {
				overrides.beginGroup(SettingCategory[CURVATURE]);
				configuration.curvature_scale.Set(overrides.value(QStringLiteral("scale"), configuration.curvature_scale.Get_raw()).toDouble());
				configuration.curvature_mode.Set(overrides.value(QStringLiteral("mode"), configuration.curvature_mode.Get_raw()).toInt());
				overrides.endGroup();
			}

			if (global_descriptor.Output[AO]) {
				overrides.beginGroup(SettingCategory[AO]);
				configuration.ao_scale.Set(overrides.value(QStringLiteral("scale"), configuration.ao_scale.Get_raw()).toDouble());
				configuration.ao_samples.Set(overrides.value(QStringLiteral("samples"), configuration.ao_samples.Get_raw()).toDouble());
				configuration.ao_distance.Set(overrides.value(QStringLiteral("distance"), configuration.ao_distance.Get_raw()).toDouble());
				configuration.ao_power.Set(overrides.value(QStringLiteral("power"), configuration.ao_power.Get_raw()).toDouble());
				overrides.endGroup();
			}

			for (int i = 0; i < NUM_OUTPUTS; i++) {
				if ((global_descriptor.Output[i]) || (global_descriptor.Input == i)) {
					overrides.beginGroup(SettingCategory[i]);
					configuration.export_settings[i].Set_format(overrides.value(QStringLiteral("format"), configuration.export_settings[i].Get_format()).toInt());
					configuration.export_settings[i].Set_bitdepth(overrides.value(QStringLiteral("bitdepth"), configuration.export_settings[i].Get_bitdepth()).toInt());
					configuration.export_settings[i].Set_suffix(overrides.value(QStringLiteral("suffix"), QString::fromStdString(configuration.export_settings[i].Get_suffix())).toString().toStdString());
					overrides.endGroup();
				}
			}
		} else {
			ABORT("Invalid settings file passed.");
		}
	}

	//GPU check
	if (global_descriptor.Output[AO]) {
		if (!CheckGPUCompute()) {
			global_descriptor.Output.set(AO, false);
			stream << QStringLiteral("[fNabla] [ERROR] No CUDA-capable GPU detected. Ambient occlusion generation has been disabled") << endl;
		}
	}

	//Process

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
			stream
				<< QStringLiteral("[fNabla][INPUT]: ")
				<< TextureSet
				<< QStringLiteral(" ")
				<< MapNames[descriptor.Input]
				<< endl;

			MeshMapArray Maps = {{
				std::make_shared<DisplacementMap>(configuration),
				std::make_shared<NormalMap>(configuration),
				std::make_shared<CurvatureMap>(configuration),
				std::make_shared<AmbientOcclusionMap>(configuration),
			}};

			cv::Mat input_image = cv::imread(file.absoluteFilePath().toStdString(), Maps[descriptor.Input]->ReadFlags);
			if (input_image.empty()) {
				stream << QStringLiteral("[fNabla][ERROR] Failed to load input. Skipping."); //we don't need to abort, we can correctly try to do the rest
			} else {
				const cv::Size shape = input_image.size();

				Maps[descriptor.Input]->Import(input_image);

				ConversionTask conversion(Maps, configuration, descriptor);
				while (!conversion.IsReady()) {
					std::this_thread::sleep_for(std::chrono::milliseconds(50)); //update at 20fps
					stream
						<< QStringLiteral("[fNabla][PROGRESS]: ")
						<< QString::fromStdString(std::to_string(int(100.0 * conversion.progress)))
						<< QStringLiteral("% (")
						<< QString::fromStdString(conversion.status)
						<< QStringLiteral(")")
						<< '\t' << '\t' << '\t' << '\t' //4 tabs to account for different sized messages
						<< '\r' << flush; //back to start of line
				}
				stream << endl; //finish progress tracker line

				QStringList MapsSaved;
				for (int i = 0; i < NUM_OUTPUTS; i++) {
					if (descriptor.Output[i]) {
						QString output = TextureSet + QString::fromStdString(configuration.export_settings[i].Get_full_suffix());
						cv::Mat save_img = Maps[i]->Export(configuration.export_settings[i].Get_CVdepth(), i != 0);
						cv::imwrite(output.toStdString(), save_img);
						MapsSaved << output;
					}
				}

				stream << QStringLiteral("[fNabla][SAVED]: ") << endl;
				for (const QString& map : MapsSaved) {
					stream << QStringLiteral("\t") << map << QStringLiteral("\n");
				}
				stream << endl;
			}
		}
	}
	stream << QStringLiteral("[fNabla][FINISHED]") << endl;
	emit Finished();
}