#pragma once

#include <QtUiPlugin/QDesignerCustomWidgetInterface>
#include <QtCore/QtPlugin>
#include "fSettingWidget.h"

#ifdef EXPORT_PLUGIN
#define API Q_DECL_EXPORT
#else
#define API
#endif

class API fSettingWidgetPlugin : public QObject, public QDesignerCustomWidgetInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "fCustomWidgetsPlugin.fSettingWidgetPlugin")
    Q_INTERFACES(QDesignerCustomWidgetInterface)

public:
    fSettingWidgetPlugin(QObject *parent = Q_NULLPTR);

    bool isContainer() const;
    bool isInitialized() const;
    QIcon icon() const;
    QString domXml() const;
    QString group() const;
    QString includeFile() const;
    QString name() const;
    QString toolTip() const;
    QString whatsThis() const;
    QWidget *createWidget(QWidget *parent);
    void initialize(QDesignerFormEditorInterface *core);

private:
    bool initialized = false;
};
