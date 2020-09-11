/****************************************************************************
** Meta object code from reading C++ file 'fSettingWidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.9)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../fNablaGUI/fSettingWidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'fSettingWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.9. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_SettingWidget_t {
    QByteArrayData data[30];
    char stringdata0[327];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_SettingWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_SettingWidget_t qt_meta_stringdata_SettingWidget = {
    {
QT_MOC_LITERAL(0, 0, 13), // "SettingWidget"
QT_MOC_LITERAL(1, 14, 12), // "valueChanged"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 5), // "value"
QT_MOC_LITERAL(4, 34, 12), // "UpdateRanges"
QT_MOC_LITERAL(5, 47, 21), // "UpdateValueFromSlider"
QT_MOC_LITERAL(6, 69, 22), // "UpdateValueFromSpinbox"
QT_MOC_LITERAL(7, 92, 28), // "UpdateValueFromDoubleSpinbox"
QT_MOC_LITERAL(8, 121, 4), // "Name"
QT_MOC_LITERAL(9, 126, 7), // "setName"
QT_MOC_LITERAL(10, 134, 7), // "newName"
QT_MOC_LITERAL(11, 142, 10), // "isDiscrete"
QT_MOC_LITERAL(12, 153, 11), // "setDiscrete"
QT_MOC_LITERAL(13, 165, 9), // "new_value"
QT_MOC_LITERAL(14, 175, 7), // "Minimum"
QT_MOC_LITERAL(15, 183, 10), // "setMinimum"
QT_MOC_LITERAL(16, 194, 7), // "Maximum"
QT_MOC_LITERAL(17, 202, 10), // "setMaximum"
QT_MOC_LITERAL(18, 213, 11), // "SingleSteps"
QT_MOC_LITERAL(19, 225, 14), // "setSingleSteps"
QT_MOC_LITERAL(20, 240, 8), // "Decimals"
QT_MOC_LITERAL(21, 249, 11), // "setDecimals"
QT_MOC_LITERAL(22, 261, 5), // "Value"
QT_MOC_LITERAL(23, 267, 8), // "setValue"
QT_MOC_LITERAL(24, 276, 8), // "decimals"
QT_MOC_LITERAL(25, 285, 11), // "singleSteps"
QT_MOC_LITERAL(26, 297, 7), // "maximum"
QT_MOC_LITERAL(27, 305, 7), // "minimum"
QT_MOC_LITERAL(28, 313, 4), // "name"
QT_MOC_LITERAL(29, 318, 8) // "discrete"

    },
    "SettingWidget\0valueChanged\0\0value\0"
    "UpdateRanges\0UpdateValueFromSlider\0"
    "UpdateValueFromSpinbox\0"
    "UpdateValueFromDoubleSpinbox\0Name\0"
    "setName\0newName\0isDiscrete\0setDiscrete\0"
    "new_value\0Minimum\0setMinimum\0Maximum\0"
    "setMaximum\0SingleSteps\0setSingleSteps\0"
    "Decimals\0setDecimals\0Value\0setValue\0"
    "decimals\0singleSteps\0maximum\0minimum\0"
    "name\0discrete"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_SettingWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      19,   14, // methods
       7,  144, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  109,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    0,  112,    2, 0x08 /* Private */,
       5,    0,  113,    2, 0x08 /* Private */,
       6,    0,  114,    2, 0x08 /* Private */,
       7,    0,  115,    2, 0x08 /* Private */,
       8,    0,  116,    2, 0x0a /* Public */,
       9,    1,  117,    2, 0x0a /* Public */,
      11,    0,  120,    2, 0x0a /* Public */,
      12,    1,  121,    2, 0x0a /* Public */,
      14,    0,  124,    2, 0x0a /* Public */,
      15,    1,  125,    2, 0x0a /* Public */,
      16,    0,  128,    2, 0x0a /* Public */,
      17,    1,  129,    2, 0x0a /* Public */,
      18,    0,  132,    2, 0x0a /* Public */,
      19,    1,  133,    2, 0x0a /* Public */,
      20,    0,  136,    2, 0x0a /* Public */,
      21,    1,  137,    2, 0x0a /* Public */,
      22,    0,  140,    2, 0x0a /* Public */,
      23,    1,  141,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double,    3,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   10,
    QMetaType::Bool,
    QMetaType::Void, QMetaType::Bool,   13,
    QMetaType::Double,
    QMetaType::Void, QMetaType::Double,   13,
    QMetaType::Double,
    QMetaType::Void, QMetaType::Double,   13,
    QMetaType::Double,
    QMetaType::Void, QMetaType::Double,   13,
    QMetaType::Int,
    QMetaType::Void, QMetaType::Int,   13,
    QMetaType::Double,
    QMetaType::Void, QMetaType::Double,   13,

 // properties: name, type, flags
      24, QMetaType::Int, 0x00095103,
      25, QMetaType::Double, 0x00095103,
       3, QMetaType::Double, 0x00095103,
      26, QMetaType::Double, 0x00095103,
      27, QMetaType::Double, 0x00095103,
      28, QMetaType::QString, 0x00095103,
      29, QMetaType::Bool, 0x00095103,

       0        // eod
};

void SettingWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        SettingWidget *_t = static_cast<SettingWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->UpdateRanges(); break;
        case 2: _t->UpdateValueFromSlider(); break;
        case 3: _t->UpdateValueFromSpinbox(); break;
        case 4: _t->UpdateValueFromDoubleSpinbox(); break;
        case 5: _t->Name(); break;
        case 6: _t->setName((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 7: { bool _r = _t->isDiscrete();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 8: _t->setDiscrete((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: { double _r = _t->Minimum();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 10: _t->setMinimum((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 11: { double _r = _t->Maximum();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 12: _t->setMaximum((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 13: { double _r = _t->SingleSteps();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 14: _t->setSingleSteps((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 15: { int _r = _t->Decimals();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 16: _t->setDecimals((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 17: { double _r = _t->Value();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 18: _t->setValue((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            typedef void (SettingWidget::*_t)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&SettingWidget::valueChanged)) {
                *result = 0;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        SettingWidget *_t = static_cast<SettingWidget *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< int*>(_v) = _t->Decimals(); break;
        case 1: *reinterpret_cast< double*>(_v) = _t->SingleSteps(); break;
        case 2: *reinterpret_cast< double*>(_v) = _t->Value(); break;
        case 3: *reinterpret_cast< double*>(_v) = _t->Maximum(); break;
        case 4: *reinterpret_cast< double*>(_v) = _t->Minimum(); break;
        case 5: *reinterpret_cast< QString*>(_v) = _t->Name(); break;
        case 6: *reinterpret_cast< bool*>(_v) = _t->isDiscrete(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        SettingWidget *_t = static_cast<SettingWidget *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setDecimals(*reinterpret_cast< int*>(_v)); break;
        case 1: _t->setSingleSteps(*reinterpret_cast< double*>(_v)); break;
        case 2: _t->setValue(*reinterpret_cast< double*>(_v)); break;
        case 3: _t->setMaximum(*reinterpret_cast< double*>(_v)); break;
        case 4: _t->setMinimum(*reinterpret_cast< double*>(_v)); break;
        case 5: _t->setName(*reinterpret_cast< QString*>(_v)); break;
        case 6: _t->setDiscrete(*reinterpret_cast< bool*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

const QMetaObject SettingWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_SettingWidget.data,
      qt_meta_data_SettingWidget,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *SettingWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *SettingWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_SettingWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int SettingWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 19)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 19;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 19)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 19;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 7;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 7;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 7;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 7;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 7;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void SettingWidget::valueChanged(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
