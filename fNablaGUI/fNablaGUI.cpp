﻿#include "fNablaGUI.h"

int main(int argc, char* argv[]){
	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("fNabla");
	QCoreApplication::setOrganizationName("fNabla");
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

	MapInfoArray = {{
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

	//CONNECTIONS
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		QObject::connect(MapInfoArray[i].ExportAction, &QAction::triggered, this, [this, i]() {
			std::bitset<NUM_OUTPUTS>map_selection;
			map_selection.set(i);
			ExportManager(map_selection);
		});
		QObject::connect(MapInfoArray[i].EnableSetting, &QCheckBox::toggled, this, [this, i](bool checked) {
			if (checked) {
				global_descriptor.Output.set(i);
			}
			else {
				global_descriptor.Output.reset(i);
			}
			if (LoadedState) {
				if (checked) {
					ui.actionExport_All->setEnabled(true);
					Descriptor descriptor{ global_descriptor.Input, std::bitset<NUM_OUTPUTS>() };
					descriptor.Output.set(i);
					ExecuteConversion(Maps, configuration, descriptor, WorkingScaleFactor);
					UpdatePixmap(i);
				}
				else {
					bool anyChecked = false;
					for (int i = 0; i < NUM_OUTPUTS; i++) {
						anyChecked = (anyChecked || MapInfoArray[i].EnableSetting->isChecked());
					}
					ui.actionExport_All->setEnabled(anyChecked);
					if (i != global_descriptor.Input)
					{
						Maps[i]->Mat.release();
					}
					MapInfoArray[i].Pixmap = DefaultImage;
				}
				if (ui.MapSelectTab->currentIndex() == i) {
					UpdateLabel();
				}
			}
		});

		QObject::connect(MapInfoArray[i].ExportDepth, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			configuration.export_settings[i].Set_bitdepth(idx);
			int updated_format = configuration.export_settings[i].Get_format();
			if (MapInfoArray[i].ExportFormat->currentIndex() != updated_format) {
				MapInfoArray[i].ExportFormat->setCurrentIndex(updated_format);
			}
		});

		QObject::connect(MapInfoArray[i].ExportFormat, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this, i](int idx) {
			configuration.export_settings[i].Set_format(idx);
			int updated_bitdepth = configuration.export_settings[i].Get_bitdepth();
			if (MapInfoArray[i].ExportDepth->currentIndex() != updated_bitdepth) {
				MapInfoArray[i].ExportDepth->setCurrentIndex(updated_bitdepth);
			}
		});

		QObject::connect(MapInfoArray[i].ExportSuffix, &QLineEdit::editingFinished, this, [this, i]() {
			configuration.export_settings[i].Set_suffix(MapInfoArray[i].ExportSuffix->text().toStdString());
		});

		QRegExpValidator SuffixValidator(QRegExp("[a-zA-Z0-9_-]{2,25}"), this);
		MapInfoArray[i].ExportSuffix->setValidator(&SuffixValidator);
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
		ui.MapSelectTab->setCurrentIndex((ui.MapSelectTab->currentIndex() + 1) % ui.MapSelectTab->count());
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
			MapInfoArray[i].Pixmap = DefaultImage;
		}
		SetLoadedState(false);
	});
	QObject::connect(ui.actionAbout_fNabla, &QAction::triggered, this, [this]() {
		QMessageBox::about(this, "About fNabla", "<b>fNabla</b><br><br>Version 1.0<br>fNabla is a tool for conversion between various mesh maps<br>Copyright (C) 2020 Borja Franco Garcia");
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
		if (LoadedState && global_descriptor.Output[DISPLACEMENT]) {
			UpdatePixmap(DISPLACEMENT);
			if (ui.MapSelectTab->currentIndex() == DISPLACEMENT) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.Settings_normal_scale, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.normal_scale.Set(value);
		if (LoadedState && global_descriptor.Output[NORMAL]) {
			UpdatePixmap(NORMAL);
			if (ui.MapSelectTab->currentIndex() == NORMAL) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.swizzle_x, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		configuration.normal_swizzle.Set(ui.swizzle_x->currentIndex(), ui.swizzle_y->currentIndex());
		if (LoadedState) {
			if (global_descriptor.Input == NORMAL) {
				ProcessInput();
				RedrawAll();
			}
			else if (global_descriptor.Output[NORMAL]) {
				ComputeMap(NORMAL);
				UpdatePixmap(NORMAL);
				if (ui.MapSelectTab->currentIndex() == NORMAL) {
					UpdateLabel();
				}
			}
		}
	});
	QObject::connect(ui.swizzle_y, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		configuration.normal_swizzle.Set(ui.swizzle_x->currentIndex(), ui.swizzle_y->currentIndex());
		if (LoadedState) {
			if (global_descriptor.Input == NORMAL) {
				ProcessInput();
				RedrawAll();
			}
			else if (global_descriptor.Output[NORMAL]) {
				ComputeMap(NORMAL);
				UpdatePixmap(NORMAL);
				if (ui.MapSelectTab->currentIndex() == NORMAL) {
					UpdateLabel();
				}
			}
		}
	});
	QObject::connect(ui.Settings_curvature_scale, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.curvature_scale.Set(value);
		if (LoadedState && global_descriptor.Output[CURVATURE]) {
			UpdatePixmap(CURVATURE);
			if (ui.MapSelectTab->currentIndex() == CURVATURE) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.Settings_curvature_mode, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int idx) {
		configuration.curvature_mode.Set(idx);
		if (LoadedState && global_descriptor.Output[CURVATURE]) {
			UpdatePixmap(CURVATURE);
			if (ui.MapSelectTab->currentIndex() == CURVATURE) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.Settings_ao_scale, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_scale.Set(value);
		if (LoadedState && global_descriptor.Output[AO]) {
			ComputeMap(AO);
			UpdatePixmap(AO);
			if (ui.MapSelectTab->currentIndex() == AO) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.Settings_ao_power, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_power.Set(value);
		if (LoadedState && global_descriptor.Output[AO]) {
			UpdatePixmap(AO);
			if (ui.MapSelectTab->currentIndex() == AO) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.Settings_ao_distance, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_distance.Set(value);
		if (LoadedState && global_descriptor.Output[AO]) {
			ComputeMap(AO);
			UpdatePixmap(AO);
			if (ui.MapSelectTab->currentIndex() == AO) {
				UpdateLabel();
			}
		}
	});
	QObject::connect(ui.Settings_ao_samples, &SettingWidget::valueChanged, this, [this](double value) {
		configuration.ao_samples.Set((int)value);
		if (LoadedState && global_descriptor.Output[AO]) {
			ComputeMap(AO);
			UpdatePixmap(AO);
			if (ui.MapSelectTab->currentIndex() == AO) {
				UpdateLabel();
			}
		}
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

//ProgressBar Macros
#define START_PROGRESSBAR(NUM_MILESTONES)		\
	int _NUM_MILESTONES = NUM_MILESTONES;		\
	int _MILESTONE_COUNTER = 0;					\
	ui.ProgressBar->setValue(0);				\


#define MILESTONE(STATUS)																\
	ui.Status->setText(QStringLiteral(STATUS));											\
	_MILESTONE_COUNTER += 1;															\
	ui.ProgressBar->setValue(int(100.0 * _MILESTONE_COUNTER/_NUM_MILESTONES));			\


void fNablaGUI::closeEvent(QCloseEvent* event) {
	settings->setValue("GUI/size", size());
	settings->setValue("GUI/pos", pos());
}

//SETTINGS

void fNablaGUI::LoadSettings() {
	settings = std::make_unique<QSettings>();

	resize(settings->value("GUI/size", QSize(1350, 900)).toSize());
	move(settings->value("GUI/pos", QPoint(100, 50)).toPoint());
	ui.WorkingResolution->setCurrentIndex(settings->value("GUI/working_resolution", 0).toInt());

	ui.Settings_high_pass->setValue(settings->value("Global/window", configuration.integration_window.Get_raw()).toDouble());

	ui.Settings_displacement_mode->setCurrentIndex(settings->value("Displacement/colormap", configuration.displacement_colormap.Get_raw()).toInt());

	ui.Settings_normal_scale->setValue(settings->value("Normal/scale", configuration.normal_scale.Get_raw()).toDouble());
	ui.swizzle_x->setCurrentIndex(settings->value("Normal/swizzle_x", configuration.normal_swizzle.Get_x()).toInt());
	ui.swizzle_y->setCurrentIndex(settings->value("Normal/swizzle_y", configuration.normal_swizzle.Get_y()).toInt());

	ui.Settings_curvature_scale->setValue(settings->value("Curvature/scale", configuration.curvature_scale.Get_raw()).toDouble());
	ui.Settings_curvature_mode->setCurrentIndex(settings->value("Curvature/mode", configuration.curvature_mode.Get_raw()).toInt());

	ui.Settings_ao_scale->setValue(settings->value("AO/scale", configuration.ao_scale.Get_raw()).toDouble());
	ui.Settings_ao_samples->setValue(settings->value("AO/samples", configuration.ao_samples.Get_raw()).toDouble());
	ui.Settings_ao_distance->setValue(settings->value("AO/distance", configuration.ao_distance.Get_raw()).toDouble());
	ui.Settings_ao_power->setValue(settings->value("AO/power", configuration.ao_power.Get_raw()).toDouble());

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		MapInfoArray[i].EnableSetting->setChecked(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/enable"), configuration.enabled_maps[i].Get()).toBool());
		MapInfoArray[i].ExportFormat->setCurrentIndex(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/format"), configuration.export_settings[i].Get_format()).toInt());
		MapInfoArray[i].ExportDepth->setCurrentIndex(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/bitdepth"), configuration.export_settings[i].Get_bitdepth()).toInt());
		MapInfoArray[i].ExportSuffix->setText(settings->value(MapInfoArray[i].SettingCategory + QStringLiteral("/suffix"), QString::fromStdString(configuration.export_settings[i].Get_suffix())).toString());
	}
}

void fNablaGUI::on_actionSaveSettings_clicked() {
	settings->setValue("GUI/working_resolution", ui.WorkingResolution->currentIndex());
	settings->setValue("Global/window", ui.Settings_high_pass->Value());

	settings->setValue("Displacement/colormap", ui.Settings_displacement_mode->currentIndex());

	settings->setValue("Normal/scale", ui.Settings_normal_scale->Value());
	settings->setValue("Normal/swizzle_x", ui.swizzle_x->currentIndex());
	settings->setValue("Normal/swizzle_y", ui.swizzle_y->currentIndex());

	settings->setValue("Curvature/scale", ui.Settings_curvature_scale->Value());
	settings->setValue("Curvature/mode", ui.Settings_curvature_mode->currentIndex());

	settings->setValue("AO/scale", ui.Settings_ao_scale->Value());
	settings->setValue("AO/samples", ui.Settings_ao_samples->Value());
	settings->setValue("AO/distance", ui.Settings_ao_distance->Value());
	settings->setValue("AO/power", ui.Settings_ao_power->Value());

	for (int i = 0; i < NUM_OUTPUTS; i++) {
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/enable"), MapInfoArray[i].EnableSetting->isChecked());
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/format"), MapInfoArray[i].ExportFormat->currentIndex());
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/bitdepth"), MapInfoArray[i].ExportDepth->currentIndex());
		settings->setValue(MapInfoArray[i].SettingCategory + QStringLiteral("/suffix"), MapInfoArray[i].ExportSuffix->text());
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
	bool anyChecked = false;
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		MapInfoArray[i].ExportAction->setEnabled(loaded);
		anyChecked = (anyChecked || MapInfoArray[i].EnableSetting->isChecked());
	}
	ui.actionExport_All->setEnabled(anyChecked && loaded);
	ui.actionFit_Window->setEnabled(loaded);
	ui.actionZoom_In->setEnabled(loaded);
	ui.actionZoom_Out->setEnabled(loaded);
	LoadedState = loaded;
}

//LOADING
void fNablaGUI::LoadManager(int i) {
	//LOADING
	QString fileName = QFileDialog::getOpenFileName(this,
		QStringLiteral("Load ") + MapInfoArray[i].Name, "",
		QStringLiteral("Image files (*.png *.tiff *.tif *.pbm);;All Files (*)"));
	if (!fileName.isEmpty()) {
		ui.Status->setText(QStringLiteral("Loading"));
		if (LoadedState)
		{
			ui.actionClear->trigger();
		}
		global_descriptor.Input = i;
		input_image = cv::imread(fileName.toStdString(), Maps[global_descriptor.Input]->ReadFlags);
		//--------------------------
		//PROCESSING
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
		MapInfoArray[i].Pixmap = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_RGB888).rgbSwapped());
	}
	else {
		MapInfoArray[i].Pixmap = QPixmap::fromImage(QImage((unsigned char*)mat_8bit.data, mat_8bit.cols, mat_8bit.rows, mat_8bit.step, QImage::Format_Grayscale8));
	}
	if (UIScaleFactor != 1.0) {
		MapInfoArray[i].Pixmap = MapInfoArray[i].Pixmap.scaled(QSize(mat_8bit.cols * UIScaleFactor, mat_8bit.rows * UIScaleFactor), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	}
}

void fNablaGUI::UpdateLabel() {
	ui.DisplayLabel->setPixmap(MapInfoArray[ui.MapSelectTab->currentIndex()].Pixmap);
	ui.DisplayLabel->adjustSize();
}

void fNablaGUI::RedrawAll() {
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		if ((global_descriptor.Input == i) || (global_descriptor.Output[i])) {
			UpdatePixmap(i);
		}
	}
	UpdateLabel();
}

//PROCESSSING

void fNablaGUI::ProcessInput(bool override_work_res) {
	double scale_factor = (override_work_res ? 1.0 : WorkingScaleFactor);

	Maps[global_descriptor.Input]->Import(input_image, scale_factor); //reload fresh unprocessed input to not accumulate

	ExecuteConversion(Maps, configuration, global_descriptor, scale_factor);
}

void fNablaGUI::ComputeMap(int i) {
	Descriptor ao_descriptor{global_descriptor.Input, std::bitset<NUM_OUTPUTS>()};
	ao_descriptor.Output.set(i);
	ExecuteConversion(Maps, configuration, ao_descriptor, WorkingScaleFactor);
}


//EXPORTING

void fNablaGUI::ExportManager(std::bitset<NUM_OUTPUTS> map_selection) {
	QString MapsSaved;
	// Asumption: All images have been processed since that happens on load. I'll have to make it so clicking this is not available until that's done processing.
	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Export texture set (suffix for map type and extension added automatically)"), "",
		tr("All Files (*)")); //we don't care about extension
	QFileInfo fileInfo(fileName);
	if (!fileName.isEmpty()) {
		std::bitset<NUM_OUTPUTS> check(map_selection);
		check &= global_descriptor.Output;
		if ((ui.WorkingResolution->currentIndex() != 0) || (check != map_selection)) //working res or told to export something we haven't processed
		{
			Descriptor descriptor{ global_descriptor.Input, map_selection };
			Maps[global_descriptor.Input]->Import(input_image, 1.0); //reload fresh unprocessed input
			ExecuteConversion(Maps, configuration, descriptor, 1.0);
		}
		for (int i = 0; i < NUM_OUTPUTS; i++) {
			if (map_selection[i]) {
				QString output = fileInfo.absolutePath() + "/" + fileInfo.baseName() + QString::fromStdString(configuration.export_settings[i].Get_full_suffix());
				cv::Mat save_img = Maps[i]->Export(configuration.export_settings[i].Get_CVdepth(), i != 0); //don't postprocess the displacement (colormap)
				if (!save_img.empty())
				{
					if (cv::imwrite(output.toStdString(), save_img)) {
						MapsSaved.append(output + QString("\n"));
					}
				}
			}
		}
		if (MapsSaved.isEmpty()) {
			QMessageBox::warning(this, "Export Failed", "No output generated");
		}
		else {
			QMessageBox::information(this, "Exported:", MapsSaved);
		}
	}
}