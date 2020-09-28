#include "fNablaGUI.h"

int main(int argc, char* argv[]){
	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("fNabla");
	QCoreApplication::setOrganizationName("BorjaFranco");
	QSettings::setDefaultFormat(QSettings::IniFormat);


	QSplashScreen splash(QPixmap(":/fNablaResources/SplashScreen.png"));
	splash.show();

	fNablaGUI window;
	window.setWindowIcon(QIcon(":/fNablaResources/fNabla.ico"));
	window.show();
	splash.finish(&window);
	return app.exec();
}

fNablaGUI::fNablaGUI(QWidget* parent) : QMainWindow(parent) {
	ui.setupUi(this);

	DefaultImage.load(":/fNablaResources/default.png");

	MapUI = {{
		{
		QStringLiteral("Displacement Map"),
		QStringLiteral("Displacement"),
		ui.Settings_displacement_suffix,
		ui.Settings_displacement_depth,
		ui.Settings_displacement_format,
		ui.actionExport_Displacement,
		ui.Settings_displacement_enable,
		DefaultImage,
		},
		{
		QStringLiteral("Normal Map"),
		QStringLiteral("Normal"),
		ui.Settings_normal_suffix,
		ui.Settings_normal_depth,
		ui.Settings_normal_format,
		ui.actionExport_Normal,
		ui.Settings_normal_enable,
		DefaultImage,
		},
		{
		QStringLiteral("Curvature Map"),
		QStringLiteral("Curvature"),
		ui.Settings_curvature_suffix,
		ui.Settings_curvature_depth,
		ui.Settings_curvature_format,
		ui.actionExport_Curvature,
		ui.Settings_curvature_enable,
		DefaultImage,
		},
		{
		QStringLiteral("Ambient Occlusion Map"),
		QStringLiteral("AO"),
		ui.Settings_ao_suffix,
		ui.Settings_ao_depth,
		ui.Settings_ao_format,
		ui.actionExport_AO,
		ui.Settings_ao_enable,
		DefaultImage,
		},
	}};

	Maps = {{
		std::make_shared<DisplacementMap>(configuration),
		std::make_shared<NormalMap>(configuration),
		std::make_shared<CurvatureMap>(configuration),
		std::make_shared<AmbientOcclusionMap>(configuration),
	}};

	HasGPU = CheckGPUCompute();
	if (!HasGPU) {
		MapUI[AO].EnableSetting->setChecked(false);
		MapUI[AO].EnableSetting->setCheckable(false);
		MapUI[AO].ExportAction->setEnabled(false);
		ui.MapSelectTab->setTabEnabled(AO, false);
		QMessageBox::warning(this, QStringLiteral("No CUDA-capable GPU detected"), QStringLiteral("Ambient occlusion generation has been disabled. Make sure you have the latest GPU drivers."));
	}

	//CONNECTIONS
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		QObject::connect(MapUI[i].ExportAction, &QAction::triggered, this, [this, i]() {
			std::bitset<NUM_OUTPUTS>map_selection;
			map_selection.set(i);
			ExportManager(map_selection);
		});
		QObject::connect(MapUI[i].EnableSetting, &QCheckBox::toggled, this, [this, i](bool checked) {
			if (checked) {
				global_descriptor.Output.set(i);
			}
			else {
				global_descriptor.Output.reset(i);
			}
			if (LoadedState) {
				MapUI[i].ExportAction->setEnabled(checked);
				if (checked) {
					ui.actionExport_All->setEnabled(true);
					ComputeMap(i);
					UpdatePixmap(i);
				}
				else {
					bool anyChecked = false;
					for (int i = 0; i < NUM_OUTPUTS; i++) {
						anyChecked = (anyChecked || MapUI[i].EnableSetting->isChecked());
					}
					ui.actionExport_All->setEnabled(anyChecked);
					if (i != global_descriptor.Input)
					{
						Maps[i]->Mat.release();
					}
					MapUI[i].Pixmap = DefaultImage;
					if (ui.MapSelectTab->currentIndex() == i) {
						ui.actionCycle_Map->trigger();
					}
				}
			}
			ui.MapSelectTab->setTabEnabled(i, checked);
		});

		QObject::connect(MapUI[i].ExportDepth, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			configuration.export_settings[i].Set_bitdepth(idx);
			if (configuration.export_settings[i].CheckCompatibility(false)) {
				MapUI[i].ExportFormat->setCurrentIndex(configuration.export_settings[i].Get_format());
			}
		});

		QObject::connect(MapUI[i].ExportFormat, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			configuration.export_settings[i].Set_format(idx);
			if (configuration.export_settings[i].CheckCompatibility(true)) {
				MapUI[i].ExportDepth->setCurrentIndex(configuration.export_settings[i].Get_bitdepth());
			}
		});

		QObject::connect(MapUI[i].ExportSuffix, &QLineEdit::editingFinished, this, [this, i]() {
			configuration.export_settings[i].Set_suffix(MapUI[i].ExportSuffix->text().toStdString());
		});

		QRegExpValidator SuffixValidator(QRegExp("[a-zA-Z0-9_-]{2,25}"), this);
		MapUI[i].ExportSuffix->setValidator(&SuffixValidator);
	}
	QObject::connect(ui.actionExport_All, &QAction::triggered, this, [this]() {
		ExportManager(global_descriptor.Output); 
	});
	QObject::connect(ui.actionZoom_In, &QAction::triggered, this, [this]() {
		Zoom(1.25);
	});
	QObject::connect(ui.actionZoom_Out, &QAction::triggered, this, [this]() {
		Zoom(0.8);
	});
	QObject::connect(ui.actionFit_Window, &QAction::triggered, this, [this]() {
		Zoom(1.0, true);
	});
	QObject::connect(ui.actionCycle_Map, &QAction::triggered, this, [this]() {
		int start = ui.MapSelectTab->currentIndex();
		for (int i = (start + 1) % ui.MapSelectTab->count(); i != start; i = (i + 1) % ui.MapSelectTab->count()) {
			if (ui.MapSelectTab->isTabEnabled(i)) {
				ui.MapSelectTab->setCurrentIndex(i);
				break;
			}
		}
	});
	QObject::connect(ui.actionLoad_Displacement, &QAction::triggered, this, [this]() {
		LoadManager(DISPLACEMENT);
	});
	QObject::connect(ui.actionLoad_TSNormal, &QAction::triggered, this, [this]() {
		LoadManager(NORMAL);
	});
	QObject::connect(ui.actionLoad_Curvature, &QAction::triggered, this, [this]() {
		LoadManager(CURVATURE);
	});
	QObject::connect(ui.actionClear, &QAction::triggered, this, [this]() {
		ui.ProgressBar->setValue(0);
		ui.Status->setText(QStringLiteral(""));
		for (int i = 0; i < NUM_OUTPUTS; i++) {
			Maps[i]->Mat.release();
			MapUI[i].Pixmap = DefaultImage;
		}
		SetLoadedState(false);
	});
	QObject::connect(ui.actionAbout_fNabla, &QAction::triggered, this, [this]() {
		QMessageBox::about(this, "About fNabla", "<b>fNabla</b><br><br>Version 1.0<br>fNabla is a tool for converting between height, normal and curvature as well as computing ambient occlusion. <br>Copyright (C) 2020 Borja Franco Garcia");
	});
	QObject::connect(ui.actionExit, &QAction::triggered, this, [this]() {
		QCoreApplication::exit(0);
	});
	QObject::connect(ui.Settings_high_pass, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.integration_window.Set(value);
		if ((LoadedState) && (global_descriptor.Input != 0)) {
			ProcessInput();
			RedrawAll();
		}
	});
	QObject::connect(ui.Settings_displacement_mode, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		configuration.displacement_colormap.Set(idx);
		MapChanged(DISPLACEMENT, false);
	});
	QObject::connect(ui.Settings_normal_scale, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.normal_scale.Set(value);
		MapChanged(NORMAL, false);
	});
	auto swizzle_slot = [this](int idx) {
		configuration.normal_swizzle.Set(ui.swizzle_x->currentIndex(), ui.swizzle_y->currentIndex());
		if (LoadedState) {
			if (global_descriptor.Input == NORMAL) {
				ProcessInput();
				RedrawAll();
			} else if (global_descriptor.Output[NORMAL]) {
				ComputeMap(NORMAL);
				UpdatePixmap(NORMAL);
				if (ui.MapSelectTab->currentIndex() == NORMAL) {
					UpdateLabel();
				}
			}
		}
	};
	QObject::connect(ui.swizzle_x, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, swizzle_slot);
	QObject::connect(ui.swizzle_y, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, swizzle_slot);

	QObject::connect(ui.Settings_curvature_scale, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.curvature_scale.Set(value);
		MapChanged(CURVATURE, false);
	});
	QObject::connect(ui.Settings_curvature_mode, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		configuration.curvature_mode.Set(idx);
		MapChanged(CURVATURE, false);
	});
	QObject::connect(ui.Settings_ao_scale, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_scale.Set(value);
		MapChanged(AO, true);
	});
	QObject::connect(ui.Settings_ao_power, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_power.Set(value);
		MapChanged(AO, false);
	});
	QObject::connect(ui.Settings_ao_distance, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_distance.Set(value);
		MapChanged(AO, true);
	});
	QObject::connect(ui.Settings_ao_samples, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_samples.Set((int)value);
		MapChanged(AO, true);
	});
	QObject::connect(ui.WorkingResolution, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		WorkingScaleFactor = exp(-log(2.0) * double(idx));

		if (LoadedState) {
			ProcessInput();
			RedrawAll();
			ui.actionFit_Window->trigger();
		}
	});

	QObject::connect(ui.MapSelectTab, &QTabWidget::currentChanged, this, [this](int idx) {
		UpdateLabel();
	});

	LoadSettings();
}

void fNablaGUI::closeEvent(QCloseEvent* event) {
	settings->setValue("GUI/size", size());
	settings->setValue("GUI/pos", pos());
}

//SETTINGS

void fNablaGUI::LoadSettings() {
	settings = std::make_unique<QSettings>();

	settings->beginGroup(QStringLiteral("GUI"));
	resize(settings->value(QStringLiteral("size"), QSize(1350, 900)).toSize());
	move(settings->value(QStringLiteral("pos"), QPoint(100, 50)).toPoint());
	ui.WorkingResolution->setCurrentIndex(settings->value(QStringLiteral("working_resolution"), 0).toInt());
	settings->endGroup();

	settings->beginGroup(QStringLiteral("Global"));
	ui.Settings_high_pass->setValue(settings->value(QStringLiteral("window"), configuration.integration_window.Get_raw()).toDouble());
	settings->endGroup();

	settings->beginGroup(MapUI[DISPLACEMENT].SettingCategory);
	ui.Settings_displacement_mode->setCurrentIndex(settings->value(QStringLiteral("colormap"), configuration.displacement_colormap.Get_raw()).toInt());
	settings->endGroup();

	settings->beginGroup(MapUI[NORMAL].SettingCategory);
	ui.Settings_normal_scale->setValue(settings->value(QStringLiteral("scale"), configuration.normal_scale.Get_raw()).toDouble());
	ui.swizzle_x->setCurrentIndex(settings->value(QStringLiteral("swizzle_x"), configuration.normal_swizzle.Get_x()).toInt());
	ui.swizzle_y->setCurrentIndex(settings->value(QStringLiteral("swizzle_y"), configuration.normal_swizzle.Get_y()).toInt());
	settings->endGroup();

	settings->beginGroup(MapUI[CURVATURE].SettingCategory);
	ui.Settings_curvature_scale->setValue(settings->value(QStringLiteral("scale"), configuration.curvature_scale.Get_raw()).toDouble());
	ui.Settings_curvature_mode->setCurrentIndex(settings->value(QStringLiteral("mode"), configuration.curvature_mode.Get_raw()).toInt());
	settings->endGroup();

	settings->beginGroup(MapUI[AO].SettingCategory);
	ui.Settings_ao_scale->setValue(settings->value(QStringLiteral("scale"), configuration.ao_scale.Get_raw()).toDouble());
	ui.Settings_ao_samples->setValue(settings->value(QStringLiteral("samples"), configuration.ao_samples.Get_raw()).toDouble());
	ui.Settings_ao_distance->setValue(settings->value(QStringLiteral("distance"), configuration.ao_distance.Get_raw()).toDouble());
	ui.Settings_ao_power->setValue(settings->value(QStringLiteral("power"), configuration.ao_power.Get_raw()).toDouble());
	settings->endGroup();

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		settings->beginGroup(MapUI[i].SettingCategory);
		MapUI[i].EnableSetting->setChecked(settings->value(QStringLiteral("enable"), configuration.enabled_maps[i].Get()).toBool());
		MapUI[i].ExportFormat->setCurrentIndex(settings->value(QStringLiteral("format"), configuration.export_settings[i].Get_format()).toInt());
		MapUI[i].ExportDepth->setCurrentIndex(settings->value(QStringLiteral("bitdepth"), configuration.export_settings[i].Get_bitdepth()).toInt());
		MapUI[i].ExportSuffix->setText(settings->value(QStringLiteral("suffix"), QString::fromStdString(configuration.export_settings[i].Get_suffix())).toString());
		settings->endGroup();
	}
}

void fNablaGUI::on_actionSaveSettings_clicked() {
	settings->beginGroup(QStringLiteral("GUI"));
	settings->setValue(QStringLiteral("working_resolution"), ui.WorkingResolution->currentIndex());
	settings->endGroup();

	settings->beginGroup(QStringLiteral("Global"));
	settings->setValue(QStringLiteral("window"), configuration.integration_window.Get_raw());
	settings->endGroup();

	settings->beginGroup(MapUI[DISPLACEMENT].SettingCategory);
	settings->setValue(QStringLiteral("colormap"), configuration.displacement_colormap.Get_raw());
	settings->endGroup();

	settings->beginGroup(MapUI[NORMAL].SettingCategory);
	settings->setValue(QStringLiteral("scale"), configuration.normal_scale.Get_raw());
	settings->setValue(QStringLiteral("swizzle_x"), configuration.normal_swizzle.Get_x());
	settings->setValue(QStringLiteral("swizzle_y"), configuration.normal_swizzle.Get_y());
	settings->endGroup();

	settings->beginGroup(MapUI[CURVATURE].SettingCategory);
	settings->setValue(QStringLiteral("scale"), configuration.curvature_scale.Get_raw());
	settings->setValue(QStringLiteral("mode"), configuration.curvature_mode.Get_raw());
	settings->endGroup();

	settings->beginGroup(MapUI[AO].SettingCategory);
	settings->setValue(QStringLiteral("scale"), configuration.ao_scale.Get_raw());
	settings->setValue(QStringLiteral("samples"), configuration.ao_samples.Get_raw());
	settings->setValue(QStringLiteral("distance"), configuration.ao_distance.Get_raw());
	settings->setValue(QStringLiteral("power"), configuration.ao_power.Get_raw());
	settings->endGroup();

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		settings->beginGroup(MapUI[i].SettingCategory);
		settings->setValue(QStringLiteral("enable"), configuration.enabled_maps[i].Get());
		settings->setValue(QStringLiteral("format"), configuration.export_settings[i].Get_format());
		settings->setValue(QStringLiteral("bitdepth"), configuration.export_settings[i].Get_bitdepth());
		settings->setValue(QStringLiteral("suffix"), QString::fromStdString(configuration.export_settings[i].Get_suffix()));
		settings->endGroup();
	}
}

//ADDITIONAL UI FUNCTIONS

void fNablaGUI::Zoom(float factor, bool fit) {
	double OldScale = UIScaleFactor;

	if (fit)
	{
		UIScaleFactor = std::min(
			(float)ui.DisplayScrollArea->height() / (float)Maps[global_descriptor.Input]->Mat.rows,
			(float)ui.DisplayScrollArea->width() / (float)Maps[global_descriptor.Input]->Mat.cols
		);
	}
	else {
		UIScaleFactor *= factor;
	}


	QPointF Pivot = ui.DisplayScrollArea->pos() + QPointF(
		ui.DisplayScrollArea->size().width() / 2.0,
		ui.DisplayScrollArea->size().height() / 2.0
	);
	QPointF ScrollbarPos = QPointF(
		ui.DisplayScrollArea->horizontalScrollBar()->value(),
		ui.DisplayScrollArea->verticalScrollBar()->value()
	);
	QPointF DeltaToPos = Pivot / OldScale - ui.DisplayScrollArea->pos() / OldScale;
	QPointF Delta = DeltaToPos * UIScaleFactor - DeltaToPos * OldScale;

	ui.DisplayScrollArea->horizontalScrollBar()->setValue(ScrollbarPos.x() + Delta.x());
	ui.DisplayScrollArea->verticalScrollBar()->setValue(ScrollbarPos.y() + Delta.y());

	RedrawAll();

	ui.actionZoom_In->setEnabled(UIScaleFactor < 10.0); //disable zoom in at 1000%
	ui.actionZoom_Out->setEnabled(UIScaleFactor > 0.1); //disable zoom out at 10%
}

void fNablaGUI::SetLoadedState(bool loaded) {
	ui.actionClear->setEnabled(loaded);
	ui.actionExport_All->setEnabled(loaded);
	bool anyEnabled = false;
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		bool enabled = MapUI[i].EnableSetting->isChecked();
		MapUI[i].ExportAction->setEnabled(loaded && enabled);
		anyEnabled = (anyEnabled || enabled);
	}
	ui.actionExport_All->setEnabled(anyEnabled && loaded);
	ui.actionFit_Window->setEnabled(loaded);
	ui.actionZoom_In->setEnabled(loaded);
	ui.actionZoom_Out->setEnabled(loaded);
	ui.MapSelectTab->setEnabled(loaded);
	ui.actionCycle_Map->setEnabled(loaded);
	LoadedState = loaded;
}

//LOADING
void fNablaGUI::LoadManager(int i) {
	//LOADING
	QString fileName = QFileDialog::getOpenFileName(this,
		QStringLiteral("Load ") + MapUI[i].Name, "",
		QStringLiteral("Image files (*.png *.tiff *.tif *.pbm);;All Files (*)"));
	if (!fileName.isEmpty()) {
		ui.Status->setText(QStringLiteral("Loading"));
		if (LoadedState)
		{
			ui.actionClear->trigger();
		}
		global_descriptor.Input = i;
		//read as-is with alpha channel and possibly wrong number of channels
		//engine's import will handle it
		input_image = cv::imread(fileName.toStdString(), cv::IMREAD_UNCHANGED | cv::IMREAD_ANYDEPTH); 
		//--------------------------
		//PROCESSING
		if (input_image.empty()) {
			QMessageBox::warning(this, QStringLiteral("Load Failed"), QStringLiteral("Invalid image format"));
			ui.actionClear->trigger();
			return;
		}
		ProcessInput();
		//-------------------
		//SET LOADED STATE
		ui.actionFit_Window->trigger();
		SetLoadedState(true);
	}
}

//DISPLAY
void fNablaGUI::UpdatePixmap(int i) {
	cv::Mat mat_8bit = Maps[i]->Export(CV_8U);

	if (mat_8bit.channels() == 3) {
		MapUI[i].Pixmap = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_RGB888).rgbSwapped());
	}
	else {
		MapUI[i].Pixmap = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_Grayscale8));
	}
	if (UIScaleFactor != 1.0) {
		MapUI[i].Pixmap = MapUI[i].Pixmap.scaled(QSize(mat_8bit.cols * UIScaleFactor, mat_8bit.rows * UIScaleFactor), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	}
}

void fNablaGUI::MapChanged(int i, bool recompute) {
	if (LoadedState && global_descriptor.Output[i]) {
		if(recompute)
			ComputeMap(i);
		UpdatePixmap(i);
		if (ui.MapSelectTab->currentIndex() == i)
			UpdateLabel();
	}
}

void fNablaGUI::UpdateLabel() {
	ui.DisplayLabel->setPixmap(MapUI[ui.MapSelectTab->currentIndex()].Pixmap);
	ui.DisplayLabel->adjustSize();
}

void fNablaGUI::RedrawAll() {
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		if ((global_descriptor.Input == i) || (global_descriptor.Output[i]))
			UpdatePixmap(i);
	}
	UpdateLabel();
}

//PROCESSSING
void fNablaGUI::MonitorProgressAndWait(ConversionTask& conversion) {

	ui.actionClear->setEnabled(false);
	ui.actionExport_All->setEnabled(false);
	bool anyEnabled = false;
	for (int i = 0; i < NUM_OUTPUTS; i++)
		MapUI[i].ExportAction->setEnabled(false);
	ui.actionExport_All->setEnabled(false);
	ui.actionLoadGroup->setEnabled(false);

	while (!conversion.CheckReady()) {
		QCoreApplication::processEvents(); //don't freeze the UI
		ui.ProgressBar->setValue(int(100.0 * conversion.progress));
		ui.Status->setText(QString::fromStdString(conversion.status));
	}
	conversion.output.get(); //"inherit" any async exceptions.
	ui.Status->setText(QStringLiteral("Ready"));

	ui.actionLoadGroup->setEnabled(true);
	SetLoadedState(true);
}

void fNablaGUI::ProcessInput(bool override_work_res) {
	double scale_factor = (override_work_res ? 1.0 : WorkingScaleFactor);

	Maps[global_descriptor.Input]->Import(input_image, scale_factor); //reload fresh unprocessed input to not accumulate
	MonitorProgressAndWait(ConversionTask(Maps, configuration, global_descriptor, scale_factor));
}

void fNablaGUI::ComputeMap(int i) {
	Descriptor descriptor;
	descriptor.Input = global_descriptor.Input;
	descriptor.Output.set(i);
	MonitorProgressAndWait(ConversionTask(Maps, configuration, descriptor, WorkingScaleFactor));
}


//EXPORTING

void fNablaGUI::ExportManager(std::bitset<NUM_OUTPUTS> map_selection) {
	QString MapsSaved;
	// Asumption: All images have been processed since that happens on load. I'll have to make it so clicking this is not available until that's done processing.
	QString fileName = QFileDialog::getSaveFileName(this,
		QStringLiteral("Export texture set (suffix for map type and extension added automatically)"), "",
		QStringLiteral("All Files (*)")); //we don't care about extension
	QFileInfo fileInfo(fileName);
	if (!fileName.isEmpty()) {
		std::bitset<NUM_OUTPUTS> check(map_selection);
		check &= global_descriptor.Output;
		if ((ui.WorkingResolution->currentIndex() != 0) || (check != map_selection)) //working res or told to export something we haven't processed
		{
			Descriptor descriptor{ global_descriptor.Input, map_selection };
			Maps[global_descriptor.Input]->Import(input_image, 1.0); //reload fresh unprocessed input
			MonitorProgressAndWait(ConversionTask(Maps, configuration, descriptor, 1.0));
		}
		for (int i = 0; i < NUM_OUTPUTS; i++) {
			if (map_selection[i]) {
				QString output = fileInfo.absolutePath() + "/" + fileInfo.baseName() + QString::fromStdString(configuration.export_settings[i].Get_full_suffix());
				cv::Mat save_img = Maps[i]->Export(configuration.export_settings[i].Get_CVdepth(), i != 0); //don't postprocess the displacement (colormap)
				if ((!save_img.empty()) && (cv::imwrite(output.toStdString(), save_img)))
					MapsSaved.append(output + QString("\n"));
			}
		}
		if (MapsSaved.isEmpty()) {
			QMessageBox::warning(this, QStringLiteral("Export Failed"), QStringLiteral("No output generated"));
		}
		else {
			QMessageBox::information(this, QStringLiteral("Exported:"), MapsSaved);
		}
	}
}