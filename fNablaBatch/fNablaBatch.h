#pragma once

#include <QtCore/QCoreApplication>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QTextStream>
#include <QSettings>
#include <QRegExpValidator>
#include <QtCore>
#include "fNablaEngine.h" //Conversion Engine, includes OpenCV

using namespace fNablaEngine;

namespace fNablaBatch
{
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


	class Task : public QObject
	{
		Q_OBJECT
	public:
		Task(QObject* parent = 0) : QObject(parent) {}

	public slots:
		void run();

	signals:
		void finished();
	};
}