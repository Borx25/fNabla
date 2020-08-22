#pragma once

#include <QtCore/QCoreApplication>
#include <QCommandLineParser>
#include <QFileInfo>
#include <QTextStream>
#include <QSettings>
#include <QRegExpValidator>
#include <QtCore>
#include "fNablaEngine.h" //Conversion Engine, includes OpenCV

namespace fNablaBatch
{
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