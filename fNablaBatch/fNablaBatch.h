#pragma once

#include <QtCore/QCoreApplication>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QTextStream>
#include <QSettings>
#include <QRegExpValidator>
#include <QDirIterator>
#include <QTimer>

#include "fNablaEngine.h" //Conversion Engine

//Opencv
#include <opencv2/highgui/highgui.hpp>

using namespace fNablaEngine;

namespace fNablaBatch
{
	class Task : public QObject
	{
		Q_OBJECT
	public:
		Task(QObject* parent = 0) : QObject(parent) {}

	public slots:
		void Run();

	signals:
		void Finished();
	};
}