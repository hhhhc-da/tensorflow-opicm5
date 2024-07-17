/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tfg::CaseOp,
::mlir::tfg::CaseRegionOp,
::mlir::tfg::ConditionOp,
::mlir::tfg::ForOp,
::mlir::tfg::ForRegionOp,
::mlir::tfg::GetResultOp,
::mlir::tfg::GraphFuncOp,
::mlir::tfg::GraphOp,
::mlir::tfg::IfOp,
::mlir::tfg::IfRegionOp,
::mlir::tfg::ReturnOp,
::mlir::tfg::StatefulCaseOp,
::mlir::tfg::StatefulCaseRegionOp,
::mlir::tfg::StatefulIfOp,
::mlir::tfg::StatefulIfRegionOp,
::mlir::tfg::StatefulWhileOp,
::mlir::tfg::StatefulWhileRegionOp,
::mlir::tfg::StatelessCaseOp,
::mlir::tfg::StatelessCaseRegionOp,
::mlir::tfg::StatelessIfOp,
::mlir::tfg::StatelessIfRegionOp,
::mlir::tfg::StatelessWhileOp,
::mlir::tfg::StatelessWhileRegionOp,
::mlir::tfg::WhileOp,
::mlir::tfg::WhileRegionOp,
::mlir::tfg::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tfg {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((::llvm::isa<::mlir::tf_type::OpaqueTensorType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be  or opaque tensor type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((::llvm::isa<::mlir::tf_type::OpaqueTensorType>(type)))) || ((::llvm::isa<::mlir::tf_type::ControlType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any tensor or control type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((::llvm::isa<::mlir::tf_type::OpaqueTensorType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of  or opaque tensor type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::tf_type::ControlType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::tf_type::ControlType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::tf_type::OpaqueTensorType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((::llvm::isa<::mlir::tf_type::OpaqueTensorType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be  or opaque tensor type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::FuncAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: an array of functions";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: type array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: An array of shapes.";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: an array of dictionaries";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tfg::RegionAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: case region preserved attributes";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::tf_type::FuncAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops6(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DictionaryAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: dictionary of named attribute values";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops6(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops7(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::tfg::RegionAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: Region attributes, argument and result attributes.";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops7(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops8(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops8(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops9(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops9(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops10(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::UnitAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: unit attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops10(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops11(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue())))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops11(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops12(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops12(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops13(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::FlatSymbolRefAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops13(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops14(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(attr))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(attr).getType().getElementType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops14(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops15(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops15(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops16(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::tf_type::VersionAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: An Attribute describing the version for a TensorFlow Graph";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops16(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops17(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: any type attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops17(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops17(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops18(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops18(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ops18(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_ops0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_ops1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::CaseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CaseOpGenericAdaptorBase::CaseOpGenericAdaptorBase(CaseOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> CaseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ArrayAttr CaseOpGenericAdaptorBase::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > CaseOpGenericAdaptorBase::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > CaseOpGenericAdaptorBase::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > CaseOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
CaseOpAdaptor::CaseOpAdaptor(CaseOp op) : CaseOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CaseOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_branches = getProperties().branches; (void)tblgen_branches;
  if (!tblgen_branches) return emitError(loc, "'tfg.Case' op ""requires attribute 'branches'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;

  if (tblgen_branches && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_branches))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_branches), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::FuncAttr>(attr))); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'branches' failed to satisfy constraint: an array of functions");

  if (tblgen_Tin && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tin))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tin), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tout))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tout), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void CaseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> CaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange CaseOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CaseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult CaseOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.Tin;
       auto attr = dict.get("Tin");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tin` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tout;
       auto attr = dict.get("Tout");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tout` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.branches;
       auto attr = dict.get("branches");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `branches` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CaseOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.Tin;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tin",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tout;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tout",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.branches;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("branches",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CaseOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.Tin.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tout.getAsOpaquePointer()), 
    llvm::hash_value(prop.branches.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CaseOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "Tin")
      return prop.Tin;

    if (name == "Tout")
      return prop.Tout;

    if (name == "branches")
      return prop.branches;

    if (name == "output_shapes")
      return prop.output_shapes;
  return std::nullopt;
}

void CaseOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "Tin") {
       prop.Tin = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tin)>>(value);
       return;
    }

    if (name == "Tout") {
       prop.Tout = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tout)>>(value);
       return;
    }

    if (name == "branches") {
       prop.branches = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.branches)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }
}

void CaseOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.Tin) attrs.append("Tin", prop.Tin);

    if (prop.Tout) attrs.append("Tout", prop.Tout);

    if (prop.branches) attrs.append("branches", prop.branches);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);
}

::mlir::LogicalResult CaseOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTinAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tin", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getToutAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tout", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBranchesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops0(attr, "branches", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult CaseOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.Tin)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tout)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.branches)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CaseOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.Tin);

  writer.writeOptionalAttribute(prop.Tout);
  writer.writeAttribute(prop.branches);

  writer.writeOptionalAttribute(prop.output_shapes);
}

::mlir::ArrayAttr CaseOp::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > CaseOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > CaseOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > CaseOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void CaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches) {
      build(odsBuilder, odsState, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void CaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().branches = branches;
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void CaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().branches = branches;
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CaseOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult CaseOp::verifyInvariantsImpl() {
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_branches = getProperties().branches; (void)tblgen_branches;
  if (!tblgen_branches) return emitOpError("requires attribute 'branches'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops0(*this, tblgen_branches, "branches")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult CaseOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyCaseLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::CaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::CaseRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CaseRegionOpGenericAdaptorBase::CaseRegionOpGenericAdaptorBase(CaseRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> CaseRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::std::optional< ::mlir::ArrayAttr > CaseRegionOpGenericAdaptorBase::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > CaseRegionOpGenericAdaptorBase::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
CaseRegionOpAdaptor::CaseRegionOpAdaptor(CaseRegionOp op) : CaseRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CaseRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_branch_attrs = getProperties().branch_attrs; (void)tblgen_branch_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (tblgen_branch_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_branch_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_branch_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'tfg.CaseRegion' op ""attribute 'branch_attrs' failed to satisfy constraint: an array of dictionaries");

  if (tblgen_region_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_region_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_region_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tfg::RegionAttr>(attr))); }))))
    return emitError(loc, "'tfg.CaseRegion' op ""attribute 'region_attrs' failed to satisfy constraint: case region preserved attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CaseRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange CaseRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CaseRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult CaseRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.branch_attrs;
       auto attr = dict.get("branch_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `branch_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.region_attrs;
       auto attr = dict.get("region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CaseRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.branch_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("branch_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("region_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CaseRegionOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.branch_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.region_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CaseRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "branch_attrs")
      return prop.branch_attrs;

    if (name == "region_attrs")
      return prop.region_attrs;
  return std::nullopt;
}

void CaseRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "branch_attrs") {
       prop.branch_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.branch_attrs)>>(value);
       return;
    }

    if (name == "region_attrs") {
       prop.region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.region_attrs)>>(value);
       return;
    }
}

void CaseRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.branch_attrs) attrs.append("branch_attrs", prop.branch_attrs);

    if (prop.region_attrs) attrs.append("region_attrs", prop.region_attrs);
}

::mlir::LogicalResult CaseRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBranchAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops3(attr, "branch_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops4(attr, "region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult CaseRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.branch_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.region_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CaseRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.branch_attrs);

  writer.writeOptionalAttribute(prop.region_attrs);
}

::std::optional< ::mlir::ArrayAttr > CaseRegionOp::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > CaseRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void CaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.getOrAddProperties<Properties>().branch_attrs = branch_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void CaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.getOrAddProperties<Properties>().branch_attrs = branch_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CaseRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CaseRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult CaseRegionOp::verifyInvariantsImpl() {
  auto tblgen_branch_attrs = getProperties().branch_attrs; (void)tblgen_branch_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_branch_attrs, "branch_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops4(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getBranches())
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "branches", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult CaseRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CaseRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand branch_indexRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> branch_indexOperands(&branch_indexRawOperand, 1);  ::llvm::SMLoc branch_indexOperandsLoc;
  (void)branch_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> branchesRegions;
  ::llvm::ArrayRef<::mlir::Type> branch_indexTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  branch_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(branch_indexRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      branchesRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        branchesRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : branchesRegions)
    if (region->empty()) region->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType branch_index__outs_functionType;
  if (parser.parseType(branch_index__outs_functionType))
    return ::mlir::failure();
  branch_indexTypes = branch_index__outs_functionType.getInputs();
  outsTypes = branch_index__outs_functionType.getResults();
  result.addRegions(branchesRegions);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(branch_indexOperands, branch_indexTypes, branch_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CaseRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBranchIndex();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
    llvm::interleaveComma(getBranches(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getBranchIndex().getType()), getOuts().getTypes());
}

StringRef CaseRegionOp::getDefaultDialect() { return "tfg"; }
void CaseRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp CaseRegionOp::branch_yield(unsigned idx) {
  return cast<YieldOp>(getBranchBlock(idx).getTerminator());
}

void CaseRegionOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpSuccessorRegions(*this, point, regions);
}
void CaseRegionOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpEntrySuccessorRegions(*this, operands, regions);
}
void CaseRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
}

LogicalResult CaseRegionOp::verify() { return VerifyCaseLikeRegionOp(*this); }

BlockArgument CaseRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument CaseRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument CaseRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument CaseRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::CaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ConditionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConditionOpGenericAdaptorBase::ConditionOpGenericAdaptorBase(ConditionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ConditionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

} // namespace detail
ConditionOpAdaptor::ConditionOpAdaptor(ConditionOp op) : ConditionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ConditionOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConditionOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange ConditionOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange ConditionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::LogicalResult ConditionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
    {

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute ConditionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.operandSegmentSizes;
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes",
                                              ::mlir::DenseI32ArrayAttr::get(ctx, propStorage)));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConditionOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> ConditionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes);
  return std::nullopt;
}

void ConditionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void ConditionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
  attrs.append("operandSegmentSizes", ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes));
}

::mlir::LogicalResult ConditionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

::mlir::LogicalResult ConditionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ConditionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

void ConditionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
}

void ConditionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConditionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ConditionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult ConditionOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConditionOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConditionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(&condRawOperand, 1);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::mlir::Type condRawType{};
  ::llvm::ArrayRef<::mlir::Type> condTypes(&condRawType, 1);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    condRawType = type;
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  }
::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(argsOperands.size()), static_cast<int32_t>(ctlsOperands.size())}), result.getOrAddProperties<ConditionOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConditionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getCond().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getArgs().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getArgs().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
  }
}

void ConditionOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ConditionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ForOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ForOpGenericAdaptorBase::ForOpGenericAdaptorBase(ForOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ForOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr ForOpGenericAdaptorBase::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional< ::mlir::ArrayAttr > ForOpGenericAdaptorBase::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
ForOpAdaptor::ForOpAdaptor(ForOp op) : ForOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ForOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitError(loc, "'tfg.For' op ""requires attribute 'body'");

  if (tblgen_body && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_body))))
    return emitError(loc, "'tfg.For' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_T && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_T))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_T), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.For' op ""attribute 'T' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

void ForOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> ForOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange ForOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ForOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult ForOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.T;
       auto attr = dict.get("T");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `T` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body;
       auto attr = dict.get("body");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ForOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.T;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("T",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ForOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.T.getAsOpaquePointer()), 
    llvm::hash_value(prop.body.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ForOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "T")
      return prop.T;

    if (name == "body")
      return prop.body;
  return std::nullopt;
}

void ForOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "T") {
       prop.T = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.T)>>(value);
       return;
    }

    if (name == "body") {
       prop.body = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body)>>(value);
       return;
    }
}

void ForOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.T) attrs.append("T", prop.T);

    if (prop.body) attrs.append("body", prop.body);
}

::mlir::LogicalResult ForOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "T", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "body", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult ForOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.T)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.body)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ForOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.T);
  writer.writeAttribute(prop.body);
}

::mlir::tf_type::FuncAttr ForOp::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional< ::mlir::ArrayAttr > ForOp::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void ForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value start, Value limit, Value delta, ValueRange args, FuncAttr body) {
      build(odsBuilder, odsState, results, start, limit, delta, args, body,
            /*T=*/{});
    
}

void ForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr body, /*optional*/::mlir::ArrayAttr T) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().body = body;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void ForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr body, /*optional*/::mlir::ArrayAttr T) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().body = body;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ForOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult ForOp::verifyInvariantsImpl() {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitOpError("requires attribute 'body'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ForOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ForRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ForRegionOpGenericAdaptorBase::ForRegionOpGenericAdaptorBase(ForRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ForRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::std::optional< ::mlir::DictionaryAttr > ForRegionOpGenericAdaptorBase::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> ForRegionOpGenericAdaptorBase::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
ForRegionOpAdaptor::ForRegionOpAdaptor(ForRegionOp op) : ForRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ForRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (tblgen_body_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_body_attrs))))
    return emitError(loc, "'tfg.ForRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_region_attrs))))
    return emitError(loc, "'tfg.ForRegion' op ""attribute 'region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ForRegionOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange ForRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange ForRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> ForRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult ForRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.body_attrs;
       auto attr = dict.get("body_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.region_attrs;
       auto attr = dict.get("region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute ForRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.body_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes",
                                              ::mlir::DenseI32ArrayAttr::get(ctx, propStorage)));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ForRegionOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.body_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.region_attrs.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> ForRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "body_attrs")
      return prop.body_attrs;

    if (name == "region_attrs")
      return prop.region_attrs;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes);
  return std::nullopt;
}

void ForRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "body_attrs") {
       prop.body_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_attrs)>>(value);
       return;
    }

    if (name == "region_attrs") {
       prop.region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.region_attrs)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void ForRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.body_attrs) attrs.append("body_attrs", prop.body_attrs);

    if (prop.region_attrs) attrs.append("region_attrs", prop.region_attrs);
  attrs.append("operandSegmentSizes", ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes));
}

::mlir::LogicalResult ForRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "body_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult ForRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.body_attrs)))
    return ::mlir::failure();

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  if (::mlir::failed(reader.readOptionalAttribute(prop.region_attrs)))
    return ::mlir::failure();

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ForRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.body_attrs);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  writer.writeOptionalAttribute(prop.region_attrs);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

::std::optional< ::mlir::DictionaryAttr > ForRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> ForRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void ForRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange init, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr region_attrs) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void ForRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange init, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr region_attrs) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ForRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult ForRegionOp::verifyInvariantsImpl() {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ForRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand startRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> startOperands(&startRawOperand, 1);  ::llvm::SMLoc startOperandsLoc;
  (void)startOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand limitRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> limitOperands(&limitRawOperand, 1);  ::llvm::SMLoc limitOperandsLoc;
  (void)limitOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand deltaRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> deltaOperands(&deltaRawOperand, 1);  ::llvm::SMLoc deltaOperandsLoc;
  (void)deltaOperandsLoc;
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::mlir::Type startRawType{};
  ::llvm::ArrayRef<::mlir::Type> startTypes(&startRawType, 1);
  ::mlir::Type limitRawType{};
  ::llvm::ArrayRef<::mlir::Type> limitTypes(&limitRawType, 1);
  ::mlir::Type deltaRawType{};
  ::llvm::ArrayRef<::mlir::Type> deltaTypes(&deltaRawType, 1);
  ::llvm::SmallVector<::mlir::Type, 1> initTypes;
  ::llvm::SmallVector<::mlir::Type, 1> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("from"))
    return ::mlir::failure();

  startOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(startRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  limitOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(limitRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("by"))
    return ::mlir::failure();

  deltaOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(deltaRawOperand))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    startRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    limitRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    deltaRawType = type;
  }
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseTypeList(initTypes))
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalArrow())) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseTypeList(outsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  result.addRegion(std::move(body_regionRegion));
::llvm::copy(::llvm::ArrayRef<int32_t>({1, 1, 1, static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}), result.getOrAddProperties<ForRegionOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(startOperands, startTypes, startOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(limitOperands, limitTypes, limitOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deltaOperands, deltaTypes, deltaOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ForRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "from";
  _odsPrinter << ' ';
  _odsPrinter << getStart();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getLimit();
  _odsPrinter << ' ' << "by";
  _odsPrinter << ' ';
  _odsPrinter << getDelta();
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getStart().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getLimit().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getDelta().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getInit().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << getInit().getTypes();
  }
  _odsPrinter << ")";
  if (!getOuts().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ' << "(";
    _odsPrinter << getOuts().getTypes();
    _odsPrinter << ")";
  }
}

StringRef ForRegionOp::getDefaultDialect() { return "tfg"; }
void ForRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp ForRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ForRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GetResultOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultOpGenericAdaptorBase::GetResultOpGenericAdaptorBase(GetResultOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef GetResultOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

uint32_t GetResultOpGenericAdaptorBase::getNumber() {
  auto attr = getNumberAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp op) : GetResultOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'tfg.get_result' op ""requires attribute 'name'");
  auto tblgen_number = getProperties().number; (void)tblgen_number;
  if (!tblgen_number) return emitError(loc, "'tfg.get_result' op ""requires attribute 'number'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'tfg.get_result' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_number && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_number))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_number).getType().isSignlessInteger(32)))))
    return emitError(loc, "'tfg.get_result' op ""attribute 'number' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::mlir::LogicalResult GetResultOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.number;
       auto attr = dict.get("number");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `number` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetResultOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.number;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("number",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetResultOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()), 
    llvm::hash_value(prop.number.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetResultOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;

    if (name == "number")
      return prop.number;
  return std::nullopt;
}

void GetResultOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }

    if (name == "number") {
       prop.number = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.number)>>(value);
       return;
    }
}

void GetResultOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);

    if (prop.number) attrs.append("number", prop.number);
}

::mlir::LogicalResult GetResultOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops8(attr, "name", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNumberAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops9(attr, "number", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult GetResultOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.number)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.name);
  writer.writeAttribute(prop.number);
}

::llvm::StringRef GetResultOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

uint32_t GetResultOp::getNumber() {
  auto attr = getNumberAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void GetResultOp::setNumber(uint32_t attrValue) {
  getProperties().number = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().number = number;
  odsState.addTypes(result);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().number = number;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(GetResultOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().number = number;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::llvm::StringRef name, uint32_t number) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().number = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), number);
  odsState.addTypes(result);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::llvm::StringRef name, uint32_t number) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().number = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), number);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(GetResultOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::llvm::StringRef name, uint32_t number) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().number = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), number);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetResultOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetResultOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(GetResultOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult GetResultOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");
  auto tblgen_number = getProperties().number; (void)tblgen_number;
  if (!tblgen_number) return emitOpError("requires attribute 'number'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops8(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops9(*this, tblgen_number, "number")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetResultOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult GetResultOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getType<::mlir::tf_type::OpaqueTensorType>();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult GetResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::mlir::IntegerAttr numberAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<GetResultOp::Properties>().name = nameAttr;
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(numberAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (numberAttr) result.getOrAddProperties<GetResultOp::Properties>().number = numberAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::OpaqueTensorType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getValue();
  _odsPrinter << ")";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNumberAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  elidedAttrs.push_back("number");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

void GetResultOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::GetResultOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GraphFuncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GraphFuncOpGenericAdaptorBase::GraphFuncOpGenericAdaptorBase(GraphFuncOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::UnitAttr GraphFuncOpGenericAdaptorBase::getGenericAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().generic);
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GraphFuncOpGenericAdaptorBase::getGeneric() {
  auto attr = getGenericAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::llvm::StringRef GraphFuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FunctionType GraphFuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::std::optional< ::mlir::ArrayAttr > GraphFuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > GraphFuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::llvm::StringRef > GraphFuncOpGenericAdaptorBase::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::UnitAttr GraphFuncOpGenericAdaptorBase::getIsStatefulAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().is_stateful);
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GraphFuncOpGenericAdaptorBase::getIsStateful() {
  auto attr = getIsStatefulAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::std::optional< ::llvm::StringRef > GraphFuncOpGenericAdaptorBase::getGradient() {
  auto attr = getGradientAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::std::optional< ::mlir::DenseIntElementsAttr > GraphFuncOpGenericAdaptorBase::getResourceArgUniqueIdsKeys() {
  auto attr = getResourceArgUniqueIdsKeysAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DenseIntElementsAttr > GraphFuncOpGenericAdaptorBase::getResourceArgUniqueIdsValues() {
  auto attr = getResourceArgUniqueIdsValuesAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > GraphFuncOpGenericAdaptorBase::getControlOutput() {
  auto attr = getControlOutputAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
GraphFuncOpAdaptor::GraphFuncOpAdaptor(GraphFuncOp op) : GraphFuncOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult GraphFuncOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_control_output = getProperties().control_output; (void)tblgen_control_output;
  auto tblgen_description = getProperties().description; (void)tblgen_description;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitError(loc, "'tfg.func' op ""requires attribute 'function_type'");
  auto tblgen_generic = getProperties().generic; (void)tblgen_generic;
  auto tblgen_gradient = getProperties().gradient; (void)tblgen_gradient;
  auto tblgen_is_stateful = getProperties().is_stateful; (void)tblgen_is_stateful;
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_resource_arg_unique_ids_keys = getProperties().resource_arg_unique_ids_keys; (void)tblgen_resource_arg_unique_ids_keys;
  auto tblgen_resource_arg_unique_ids_values = getProperties().resource_arg_unique_ids_values; (void)tblgen_resource_arg_unique_ids_values;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitError(loc, "'tfg.func' op ""requires attribute 'sym_name'");

  if (tblgen_generic && !((::llvm::isa<::mlir::UnitAttr>(tblgen_generic))))
    return emitError(loc, "'tfg.func' op ""attribute 'generic' failed to satisfy constraint: unit attribute");

  if (tblgen_sym_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_sym_name))))
    return emitError(loc, "'tfg.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_function_type))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue())))))
    return emitError(loc, "'tfg.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_arg_attrs && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_arg_attrs))))
    return emitError(loc, "'tfg.func' op ""attribute 'arg_attrs' failed to satisfy constraint: array attribute");

  if (tblgen_res_attrs && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_res_attrs))))
    return emitError(loc, "'tfg.func' op ""attribute 'res_attrs' failed to satisfy constraint: array attribute");

  if (tblgen_description && !((::llvm::isa<::mlir::StringAttr>(tblgen_description))))
    return emitError(loc, "'tfg.func' op ""attribute 'description' failed to satisfy constraint: string attribute");

  if (tblgen_is_stateful && !((::llvm::isa<::mlir::UnitAttr>(tblgen_is_stateful))))
    return emitError(loc, "'tfg.func' op ""attribute 'is_stateful' failed to satisfy constraint: unit attribute");

  if (tblgen_gradient && !((::llvm::isa<::mlir::FlatSymbolRefAttr>(tblgen_gradient))))
    return emitError(loc, "'tfg.func' op ""attribute 'gradient' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_resource_arg_unique_ids_keys && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_resource_arg_unique_ids_keys))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_resource_arg_unique_ids_keys).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'tfg.func' op ""attribute 'resource_arg_unique_ids_keys' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_resource_arg_unique_ids_values && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_resource_arg_unique_ids_values))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_resource_arg_unique_ids_values).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'tfg.func' op ""attribute 'resource_arg_unique_ids_values' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_control_output && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_control_output))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_control_output), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'tfg.func' op ""attribute 'control_output' failed to satisfy constraint: string array attribute");
  return ::mlir::success();
}

::mlir::LogicalResult GraphFuncOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.arg_attrs;
       auto attr = dict.get("arg_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `arg_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.control_output;
       auto attr = dict.get("control_output");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `control_output` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.description;
       auto attr = dict.get("description");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `description` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.function_type;
       auto attr = dict.get("function_type");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `function_type` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.generic;
       auto attr = dict.get("generic");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `generic` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.gradient;
       auto attr = dict.get("gradient");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `gradient` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.is_stateful;
       auto attr = dict.get("is_stateful");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `is_stateful` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.res_attrs;
       auto attr = dict.get("res_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `res_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.resource_arg_unique_ids_keys;
       auto attr = dict.get("resource_arg_unique_ids_keys");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `resource_arg_unique_ids_keys` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.resource_arg_unique_ids_values;
       auto attr = dict.get("resource_arg_unique_ids_values");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `resource_arg_unique_ids_values` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sym_name;
       auto attr = dict.get("sym_name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sym_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GraphFuncOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.arg_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("arg_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.control_output;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("control_output",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.description;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("description",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.function_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("function_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.generic;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("generic",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.gradient;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("gradient",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.is_stateful;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("is_stateful",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.res_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("res_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.resource_arg_unique_ids_keys;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("resource_arg_unique_ids_keys",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.resource_arg_unique_ids_values;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("resource_arg_unique_ids_values",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sym_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sym_name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GraphFuncOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.arg_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.control_output.getAsOpaquePointer()), 
    llvm::hash_value(prop.description.getAsOpaquePointer()), 
    llvm::hash_value(prop.function_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.generic.getAsOpaquePointer()), 
    llvm::hash_value(prop.gradient.getAsOpaquePointer()), 
    llvm::hash_value(prop.is_stateful.getAsOpaquePointer()), 
    llvm::hash_value(prop.res_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.resource_arg_unique_ids_keys.getAsOpaquePointer()), 
    llvm::hash_value(prop.resource_arg_unique_ids_values.getAsOpaquePointer()), 
    llvm::hash_value(prop.sym_name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GraphFuncOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "arg_attrs")
      return prop.arg_attrs;

    if (name == "control_output")
      return prop.control_output;

    if (name == "description")
      return prop.description;

    if (name == "function_type")
      return prop.function_type;

    if (name == "generic")
      return prop.generic;

    if (name == "gradient")
      return prop.gradient;

    if (name == "is_stateful")
      return prop.is_stateful;

    if (name == "res_attrs")
      return prop.res_attrs;

    if (name == "resource_arg_unique_ids_keys")
      return prop.resource_arg_unique_ids_keys;

    if (name == "resource_arg_unique_ids_values")
      return prop.resource_arg_unique_ids_values;

    if (name == "sym_name")
      return prop.sym_name;
  return std::nullopt;
}

void GraphFuncOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "arg_attrs") {
       prop.arg_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.arg_attrs)>>(value);
       return;
    }

    if (name == "control_output") {
       prop.control_output = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.control_output)>>(value);
       return;
    }

    if (name == "description") {
       prop.description = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.description)>>(value);
       return;
    }

    if (name == "function_type") {
       prop.function_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.function_type)>>(value);
       return;
    }

    if (name == "generic") {
       prop.generic = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.generic)>>(value);
       return;
    }

    if (name == "gradient") {
       prop.gradient = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.gradient)>>(value);
       return;
    }

    if (name == "is_stateful") {
       prop.is_stateful = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.is_stateful)>>(value);
       return;
    }

    if (name == "res_attrs") {
       prop.res_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.res_attrs)>>(value);
       return;
    }

    if (name == "resource_arg_unique_ids_keys") {
       prop.resource_arg_unique_ids_keys = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.resource_arg_unique_ids_keys)>>(value);
       return;
    }

    if (name == "resource_arg_unique_ids_values") {
       prop.resource_arg_unique_ids_values = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.resource_arg_unique_ids_values)>>(value);
       return;
    }

    if (name == "sym_name") {
       prop.sym_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sym_name)>>(value);
       return;
    }
}

void GraphFuncOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.arg_attrs) attrs.append("arg_attrs", prop.arg_attrs);

    if (prop.control_output) attrs.append("control_output", prop.control_output);

    if (prop.description) attrs.append("description", prop.description);

    if (prop.function_type) attrs.append("function_type", prop.function_type);

    if (prop.generic) attrs.append("generic", prop.generic);

    if (prop.gradient) attrs.append("gradient", prop.gradient);

    if (prop.is_stateful) attrs.append("is_stateful", prop.is_stateful);

    if (prop.res_attrs) attrs.append("res_attrs", prop.res_attrs);

    if (prop.resource_arg_unique_ids_keys) attrs.append("resource_arg_unique_ids_keys", prop.resource_arg_unique_ids_keys);

    if (prop.resource_arg_unique_ids_values) attrs.append("resource_arg_unique_ids_values", prop.resource_arg_unique_ids_values);

    if (prop.sym_name) attrs.append("sym_name", prop.sym_name);
}

::mlir::LogicalResult GraphFuncOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getArgAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops12(attr, "arg_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getControlOutputAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops15(attr, "control_output", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getDescriptionAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops8(attr, "description", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getFunctionTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops11(attr, "function_type", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getGenericAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops10(attr, "generic", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getGradientAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops13(attr, "gradient", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIsStatefulAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops10(attr, "is_stateful", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getResAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops12(attr, "res_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getResourceArgUniqueIdsKeysAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops14(attr, "resource_arg_unique_ids_keys", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getResourceArgUniqueIdsValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops14(attr, "resource_arg_unique_ids_values", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSymNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops8(attr, "sym_name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult GraphFuncOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.arg_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.control_output)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.description)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.function_type)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.generic)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.gradient)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.is_stateful)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.res_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.resource_arg_unique_ids_keys)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.resource_arg_unique_ids_values)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.sym_name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GraphFuncOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.arg_attrs);

  writer.writeOptionalAttribute(prop.control_output);

  writer.writeOptionalAttribute(prop.description);
  writer.writeAttribute(prop.function_type);

  writer.writeOptionalAttribute(prop.generic);

  writer.writeOptionalAttribute(prop.gradient);

  writer.writeOptionalAttribute(prop.is_stateful);

  writer.writeOptionalAttribute(prop.res_attrs);

  writer.writeOptionalAttribute(prop.resource_arg_unique_ids_keys);

  writer.writeOptionalAttribute(prop.resource_arg_unique_ids_values);
  writer.writeAttribute(prop.sym_name);
}

bool GraphFuncOp::getGeneric() {
  auto attr = getGenericAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::llvm::StringRef GraphFuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FunctionType GraphFuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::std::optional< ::mlir::ArrayAttr > GraphFuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > GraphFuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::llvm::StringRef > GraphFuncOp::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

bool GraphFuncOp::getIsStateful() {
  auto attr = getIsStatefulAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::std::optional< ::llvm::StringRef > GraphFuncOp::getGradient() {
  auto attr = getGradientAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::std::optional< ::mlir::DenseIntElementsAttr > GraphFuncOp::getResourceArgUniqueIdsKeys() {
  auto attr = getResourceArgUniqueIdsKeysAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DenseIntElementsAttr > GraphFuncOp::getResourceArgUniqueIdsValues() {
  auto attr = getResourceArgUniqueIdsValuesAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > GraphFuncOp::getControlOutput() {
  auto attr = getControlOutputAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void GraphFuncOp::setGeneric(bool attrValue) {
    auto &odsProp = getProperties().generic;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void GraphFuncOp::setSymName(::llvm::StringRef attrValue) {
  getProperties().sym_name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void GraphFuncOp::setFunctionType(::mlir::FunctionType attrValue) {
  getProperties().function_type = ::mlir::TypeAttr::get(attrValue);
}

void GraphFuncOp::setDescription(::std::optional<::llvm::StringRef> attrValue) {
    auto &odsProp = getProperties().description;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue);
    else
      odsProp = nullptr;
}

void GraphFuncOp::setIsStateful(bool attrValue) {
    auto &odsProp = getProperties().is_stateful;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void GraphFuncOp::setGradient(::std::optional<::llvm::StringRef> attrValue) {
    auto &odsProp = getProperties().gradient;
    if (attrValue)
      odsProp = ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), *attrValue);
    else
      odsProp = nullptr;
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 (void)odsState.addRegion(); 
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, const Twine &sym_name, FunctionType type, bool generic) {
      build(odsBuilder, odsState);
      odsState.addAttribute(GraphFuncOp::getSymNameAttrName(odsState.name),
                           odsBuilder.getStringAttr(sym_name));
      odsState.addAttribute(GraphFuncOp::getFunctionTypeAttrName(odsState.name),
                           TypeAttr::get(type));
      if (generic) {
        odsState.addAttribute(GraphFuncOp::getGenericAttrName(odsState.name),
                             odsBuilder.getUnitAttr());
      }
    
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr generic, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/::mlir::UnitAttr is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.getOrAddProperties<Properties>().generic = generic;
  }
  odsState.getOrAddProperties<Properties>().sym_name = sym_name;
  odsState.getOrAddProperties<Properties>().function_type = function_type;
  if (arg_attrs) {
    odsState.getOrAddProperties<Properties>().arg_attrs = arg_attrs;
  }
  if (res_attrs) {
    odsState.getOrAddProperties<Properties>().res_attrs = res_attrs;
  }
  if (description) {
    odsState.getOrAddProperties<Properties>().description = description;
  }
  if (is_stateful) {
    odsState.getOrAddProperties<Properties>().is_stateful = is_stateful;
  }
  if (gradient) {
    odsState.getOrAddProperties<Properties>().gradient = gradient;
  }
  if (resource_arg_unique_ids_keys) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_keys = resource_arg_unique_ids_keys;
  }
  if (resource_arg_unique_ids_values) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_values = resource_arg_unique_ids_values;
  }
  if (control_output) {
    odsState.getOrAddProperties<Properties>().control_output = control_output;
  }
  (void)odsState.addRegion();
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr generic, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/::mlir::UnitAttr is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.getOrAddProperties<Properties>().generic = generic;
  }
  odsState.getOrAddProperties<Properties>().sym_name = sym_name;
  odsState.getOrAddProperties<Properties>().function_type = function_type;
  if (arg_attrs) {
    odsState.getOrAddProperties<Properties>().arg_attrs = arg_attrs;
  }
  if (res_attrs) {
    odsState.getOrAddProperties<Properties>().res_attrs = res_attrs;
  }
  if (description) {
    odsState.getOrAddProperties<Properties>().description = description;
  }
  if (is_stateful) {
    odsState.getOrAddProperties<Properties>().is_stateful = is_stateful;
  }
  if (gradient) {
    odsState.getOrAddProperties<Properties>().gradient = gradient;
  }
  if (resource_arg_unique_ids_keys) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_keys = resource_arg_unique_ids_keys;
  }
  if (resource_arg_unique_ids_values) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_values = resource_arg_unique_ids_values;
  }
  if (control_output) {
    odsState.getOrAddProperties<Properties>().control_output = control_output;
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool generic, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/bool is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.getOrAddProperties<Properties>().generic = ((generic) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.getOrAddProperties<Properties>().sym_name = odsBuilder.getStringAttr(sym_name);
  odsState.getOrAddProperties<Properties>().function_type = ::mlir::TypeAttr::get(function_type);
  if (arg_attrs) {
    odsState.getOrAddProperties<Properties>().arg_attrs = arg_attrs;
  }
  if (res_attrs) {
    odsState.getOrAddProperties<Properties>().res_attrs = res_attrs;
  }
  if (description) {
    odsState.getOrAddProperties<Properties>().description = description;
  }
  if (is_stateful) {
    odsState.getOrAddProperties<Properties>().is_stateful = ((is_stateful) ? odsBuilder.getUnitAttr() : nullptr);
  }
  if (gradient) {
    odsState.getOrAddProperties<Properties>().gradient = gradient;
  }
  if (resource_arg_unique_ids_keys) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_keys = resource_arg_unique_ids_keys;
  }
  if (resource_arg_unique_ids_values) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_values = resource_arg_unique_ids_values;
  }
  if (control_output) {
    odsState.getOrAddProperties<Properties>().control_output = control_output;
  }
  (void)odsState.addRegion();
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool generic, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/bool is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.getOrAddProperties<Properties>().generic = ((generic) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.getOrAddProperties<Properties>().sym_name = odsBuilder.getStringAttr(sym_name);
  odsState.getOrAddProperties<Properties>().function_type = ::mlir::TypeAttr::get(function_type);
  if (arg_attrs) {
    odsState.getOrAddProperties<Properties>().arg_attrs = arg_attrs;
  }
  if (res_attrs) {
    odsState.getOrAddProperties<Properties>().res_attrs = res_attrs;
  }
  if (description) {
    odsState.getOrAddProperties<Properties>().description = description;
  }
  if (is_stateful) {
    odsState.getOrAddProperties<Properties>().is_stateful = ((is_stateful) ? odsBuilder.getUnitAttr() : nullptr);
  }
  if (gradient) {
    odsState.getOrAddProperties<Properties>().gradient = gradient;
  }
  if (resource_arg_unique_ids_keys) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_keys = resource_arg_unique_ids_keys;
  }
  if (resource_arg_unique_ids_values) {
    odsState.getOrAddProperties<Properties>().resource_arg_unique_ids_values = resource_arg_unique_ids_values;
  }
  if (control_output) {
    odsState.getOrAddProperties<Properties>().control_output = control_output;
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GraphFuncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GraphFuncOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult GraphFuncOp::verifyInvariantsImpl() {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_control_output = getProperties().control_output; (void)tblgen_control_output;
  auto tblgen_description = getProperties().description; (void)tblgen_description;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitOpError("requires attribute 'function_type'");
  auto tblgen_generic = getProperties().generic; (void)tblgen_generic;
  auto tblgen_gradient = getProperties().gradient; (void)tblgen_gradient;
  auto tblgen_is_stateful = getProperties().is_stateful; (void)tblgen_is_stateful;
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_resource_arg_unique_ids_keys = getProperties().resource_arg_unique_ids_keys; (void)tblgen_resource_arg_unique_ids_keys;
  auto tblgen_resource_arg_unique_ids_values = getProperties().resource_arg_unique_ids_values; (void)tblgen_resource_arg_unique_ids_values;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitOpError("requires attribute 'sym_name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops10(*this, tblgen_generic, "generic")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops8(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops11(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops12(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops12(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops8(*this, tblgen_description, "description")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops10(*this, tblgen_is_stateful, "is_stateful")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops13(*this, tblgen_gradient, "gradient")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops14(*this, tblgen_resource_arg_unique_ids_keys, "resource_arg_unique_ids_keys")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops14(*this, tblgen_resource_arg_unique_ids_values, "resource_arg_unique_ids_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops15(*this, tblgen_control_output, "control_output")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GraphFuncOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void GraphFuncOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::GraphFuncOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GraphOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GraphOpGenericAdaptorBase::GraphOpGenericAdaptorBase(GraphOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::tf_type::VersionAttr GraphOpGenericAdaptorBase::getVersion() {
  auto attr = getVersionAttr();
  return ::llvm::cast<::mlir::tf_type::VersionAttr>(attr);
}

} // namespace detail
GraphOpAdaptor::GraphOpAdaptor(GraphOp op) : GraphOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult GraphOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_version = getProperties().version; (void)tblgen_version;
  if (!tblgen_version) return emitError(loc, "'tfg.graph' op ""requires attribute 'version'");

  if (tblgen_version && !((::llvm::isa<::mlir::tf_type::VersionAttr>(tblgen_version))))
    return emitError(loc, "'tfg.graph' op ""attribute 'version' failed to satisfy constraint: An Attribute describing the version for a TensorFlow Graph");
  return ::mlir::success();
}

::mlir::LogicalResult GraphOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.version;
       auto attr = dict.get("version");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `version` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GraphOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.version;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("version",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GraphOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.version.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GraphOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "version")
      return prop.version;
  return std::nullopt;
}

void GraphOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "version") {
       prop.version = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.version)>>(value);
       return;
    }
}

void GraphOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.version) attrs.append("version", prop.version);
}

::mlir::LogicalResult GraphOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getVersionAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops16(attr, "version", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult GraphOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.version)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GraphOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.version);
}

::mlir::tf_type::VersionAttr GraphOp::getVersion() {
  auto attr = getVersionAttr();
  return ::llvm::cast<::mlir::tf_type::VersionAttr>(attr);
}

void GraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::tf_type::VersionAttr version) {
  odsState.getOrAddProperties<Properties>().version = version;
  (void)odsState.addRegion();
}

void GraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::tf_type::VersionAttr version) {
  odsState.getOrAddProperties<Properties>().version = version;
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GraphOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GraphOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult GraphOp::verifyInvariantsImpl() {
  auto tblgen_version = getProperties().version; (void)tblgen_version;
  if (!tblgen_version) return emitOpError("requires attribute 'version'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops16(*this, tblgen_version, "version")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "nodes", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GraphOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GraphOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::tf_type::VersionAttr versionAttr;
  std::unique_ptr<::mlir::Region> nodesRegion = std::make_unique<::mlir::Region>();

  if (parser.parseAttribute(versionAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (versionAttr) result.getOrAddProperties<GraphOp::Properties>().version = versionAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }

  if (parser.parseRegion(*nodesRegion))
    return ::mlir::failure();

  if (nodesRegion->empty()) nodesRegion->emplaceBlock();
  result.addRegion(std::move(nodesRegion));
  return ::mlir::success();
}

void GraphOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getVersionAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("version");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getNodes());
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::GraphOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::IfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
IfOpGenericAdaptorBase::IfOpGenericAdaptorBase(IfOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> IfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr IfOpGenericAdaptorBase::getThenBranch() {
  auto attr = getThenBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr IfOpGenericAdaptorBase::getElseBranch() {
  auto attr = getElseBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional<::mlir::Type> IfOpGenericAdaptorBase::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::std::optional<::mlir::Type>(::llvm::cast<::mlir::Type>(attr.getValue())) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > IfOpGenericAdaptorBase::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > IfOpGenericAdaptorBase::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > IfOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
IfOpAdaptor::IfOpAdaptor(IfOp op) : IfOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult IfOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_Tcond = getProperties().Tcond; (void)tblgen_Tcond;
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_else_branch = getProperties().else_branch; (void)tblgen_else_branch;
  if (!tblgen_else_branch) return emitError(loc, "'tfg.If' op ""requires attribute 'else_branch'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_then_branch = getProperties().then_branch; (void)tblgen_then_branch;
  if (!tblgen_then_branch) return emitError(loc, "'tfg.If' op ""requires attribute 'then_branch'");

  if (tblgen_then_branch && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_then_branch))))
    return emitError(loc, "'tfg.If' op ""attribute 'then_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_else_branch && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_else_branch))))
    return emitError(loc, "'tfg.If' op ""attribute 'else_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_Tcond && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_Tcond))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_Tcond).getValue()))) && ((true))))
    return emitError(loc, "'tfg.If' op ""attribute 'Tcond' failed to satisfy constraint: any type attribute");

  if (tblgen_Tin && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tin))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tin), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.If' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tout))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tout), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.If' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.If' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void IfOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> IfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange IfOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult IfOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.Tcond;
       auto attr = dict.get("Tcond");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tcond` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tin;
       auto attr = dict.get("Tin");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tin` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tout;
       auto attr = dict.get("Tout");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tout` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.else_branch;
       auto attr = dict.get("else_branch");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_branch` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_branch;
       auto attr = dict.get("then_branch");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_branch` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute IfOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.Tcond;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tcond",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tin;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tin",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tout;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tout",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.else_branch;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_branch",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_branch;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_branch",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code IfOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.Tcond.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tin.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tout.getAsOpaquePointer()), 
    llvm::hash_value(prop.else_branch.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_branch.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> IfOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "Tcond")
      return prop.Tcond;

    if (name == "Tin")
      return prop.Tin;

    if (name == "Tout")
      return prop.Tout;

    if (name == "else_branch")
      return prop.else_branch;

    if (name == "output_shapes")
      return prop.output_shapes;

    if (name == "then_branch")
      return prop.then_branch;
  return std::nullopt;
}

void IfOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "Tcond") {
       prop.Tcond = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tcond)>>(value);
       return;
    }

    if (name == "Tin") {
       prop.Tin = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tin)>>(value);
       return;
    }

    if (name == "Tout") {
       prop.Tout = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tout)>>(value);
       return;
    }

    if (name == "else_branch") {
       prop.else_branch = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_branch)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }

    if (name == "then_branch") {
       prop.then_branch = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_branch)>>(value);
       return;
    }
}

void IfOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.Tcond) attrs.append("Tcond", prop.Tcond);

    if (prop.Tin) attrs.append("Tin", prop.Tin);

    if (prop.Tout) attrs.append("Tout", prop.Tout);

    if (prop.else_branch) attrs.append("else_branch", prop.else_branch);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);

    if (prop.then_branch) attrs.append("then_branch", prop.then_branch);
}

::mlir::LogicalResult IfOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTcondAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops17(attr, "Tcond", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTinAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tin", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getToutAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tout", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getElseBranchAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "else_branch", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenBranchAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "then_branch", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult IfOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.Tcond)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tin)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tout)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.else_branch)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.then_branch)))
    return ::mlir::failure();
  return ::mlir::success();
}

void IfOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.Tcond);

  writer.writeOptionalAttribute(prop.Tin);

  writer.writeOptionalAttribute(prop.Tout);
  writer.writeAttribute(prop.else_branch);

  writer.writeOptionalAttribute(prop.output_shapes);
  writer.writeAttribute(prop.then_branch);
}

::mlir::tf_type::FuncAttr IfOp::getThenBranch() {
  auto attr = getThenBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr IfOp::getElseBranch() {
  auto attr = getElseBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional<::mlir::Type> IfOp::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::std::optional<::mlir::Type>(::llvm::cast<::mlir::Type>(attr.getValue())) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > IfOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > IfOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > IfOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void IfOp::setTcond(::std::optional<::mlir::Type> attrValue) {
    auto &odsProp = getProperties().Tcond;
    if (attrValue)
      odsProp = ::mlir::TypeAttr::get(*attrValue);
    else
      odsProp = nullptr;
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch) {
      build(odsBuilder, odsState, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().then_branch = then_branch;
  odsState.getOrAddProperties<Properties>().else_branch = else_branch;
  if (Tcond) {
    odsState.getOrAddProperties<Properties>().Tcond = Tcond;
  }
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().then_branch = then_branch;
  odsState.getOrAddProperties<Properties>().else_branch = else_branch;
  if (Tcond) {
    odsState.getOrAddProperties<Properties>().Tcond = Tcond;
  }
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<IfOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult IfOp::verifyInvariantsImpl() {
  auto tblgen_Tcond = getProperties().Tcond; (void)tblgen_Tcond;
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_else_branch = getProperties().else_branch; (void)tblgen_else_branch;
  if (!tblgen_else_branch) return emitOpError("requires attribute 'else_branch'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_then_branch = getProperties().then_branch; (void)tblgen_then_branch;
  if (!tblgen_then_branch) return emitOpError("requires attribute 'then_branch'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_then_branch, "then_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_else_branch, "else_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops17(*this, tblgen_Tcond, "Tcond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult IfOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyIfLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::IfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::IfRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
IfRegionOpGenericAdaptorBase::IfRegionOpGenericAdaptorBase(IfRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> IfRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::std::optional< ::mlir::DictionaryAttr > IfRegionOpGenericAdaptorBase::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > IfRegionOpGenericAdaptorBase::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> IfRegionOpGenericAdaptorBase::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> IfRegionOpGenericAdaptorBase::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
IfRegionOpAdaptor::IfRegionOpAdaptor(IfRegionOp op) : IfRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult IfRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_else_attrs = getProperties().else_attrs; (void)tblgen_else_attrs;
  auto tblgen_else_region_attrs = getProperties().else_region_attrs; (void)tblgen_else_region_attrs;
  auto tblgen_then_attrs = getProperties().then_attrs; (void)tblgen_then_attrs;
  auto tblgen_then_region_attrs = getProperties().then_region_attrs; (void)tblgen_then_region_attrs;

  if (tblgen_then_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_then_attrs))))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'then_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_else_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_else_attrs))))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'else_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_then_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_then_region_attrs))))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'then_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_else_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_else_region_attrs))))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'else_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IfRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange IfRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IfRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult IfRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.else_attrs;
       auto attr = dict.get("else_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.else_region_attrs;
       auto attr = dict.get("else_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_attrs;
       auto attr = dict.get("then_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_region_attrs;
       auto attr = dict.get("then_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute IfRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.else_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.else_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_region_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code IfRegionOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.else_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.else_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_region_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> IfRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "else_attrs")
      return prop.else_attrs;

    if (name == "else_region_attrs")
      return prop.else_region_attrs;

    if (name == "then_attrs")
      return prop.then_attrs;

    if (name == "then_region_attrs")
      return prop.then_region_attrs;
  return std::nullopt;
}

void IfRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "else_attrs") {
       prop.else_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_attrs)>>(value);
       return;
    }

    if (name == "else_region_attrs") {
       prop.else_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_region_attrs)>>(value);
       return;
    }

    if (name == "then_attrs") {
       prop.then_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_attrs)>>(value);
       return;
    }

    if (name == "then_region_attrs") {
       prop.then_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_region_attrs)>>(value);
       return;
    }
}

void IfRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.else_attrs) attrs.append("else_attrs", prop.else_attrs);

    if (prop.else_region_attrs) attrs.append("else_region_attrs", prop.else_region_attrs);

    if (prop.then_attrs) attrs.append("then_attrs", prop.then_attrs);

    if (prop.then_region_attrs) attrs.append("then_region_attrs", prop.then_region_attrs);
}

::mlir::LogicalResult IfRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getElseAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "else_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getElseRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "else_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "then_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "then_region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult IfRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.else_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.else_region_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.then_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.then_region_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void IfRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.else_attrs);

  writer.writeOptionalAttribute(prop.else_region_attrs);

  writer.writeOptionalAttribute(prop.then_attrs);

  writer.writeOptionalAttribute(prop.then_region_attrs);
}

::std::optional< ::mlir::DictionaryAttr > IfRegionOp::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > IfRegionOp::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> IfRegionOp::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> IfRegionOp::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void IfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.getOrAddProperties<Properties>().then_attrs = then_attrs;
  }
  if (else_attrs) {
    odsState.getOrAddProperties<Properties>().else_attrs = else_attrs;
  }
  if (then_region_attrs) {
    odsState.getOrAddProperties<Properties>().then_region_attrs = then_region_attrs;
  }
  if (else_region_attrs) {
    odsState.getOrAddProperties<Properties>().else_region_attrs = else_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void IfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.getOrAddProperties<Properties>().then_attrs = then_attrs;
  }
  if (else_attrs) {
    odsState.getOrAddProperties<Properties>().else_attrs = else_attrs;
  }
  if (then_region_attrs) {
    odsState.getOrAddProperties<Properties>().then_region_attrs = then_region_attrs;
  }
  if (else_region_attrs) {
    odsState.getOrAddProperties<Properties>().else_region_attrs = else_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IfRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<IfRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult IfRegionOp::verifyInvariantsImpl() {
  auto tblgen_else_attrs = getProperties().else_attrs; (void)tblgen_else_attrs;
  auto tblgen_else_region_attrs = getProperties().else_region_attrs; (void)tblgen_else_region_attrs;
  auto tblgen_then_attrs = getProperties().then_attrs; (void)tblgen_then_attrs;
  auto tblgen_then_region_attrs = getProperties().then_region_attrs; (void)tblgen_then_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_then_attrs, "then_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_else_attrs, "else_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_then_region_attrs, "then_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_else_region_attrs, "else_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "then_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "else_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult IfRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult IfRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(&condRawOperand, 1);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> then_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> else_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> condTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("then"))
    return ::mlir::failure();

  if (parser.parseRegion(*then_regionRegion))
    return ::mlir::failure();

  if (then_regionRegion->empty()) then_regionRegion->emplaceBlock();
  if (parser.parseKeyword("else"))
    return ::mlir::failure();

  if (parser.parseRegion(*else_regionRegion))
    return ::mlir::failure();

  if (else_regionRegion->empty()) else_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType cond__outs_functionType;
  if (parser.parseType(cond__outs_functionType))
    return ::mlir::failure();
  condTypes = cond__outs_functionType.getInputs();
  outsTypes = cond__outs_functionType.getResults();
  result.addRegion(std::move(then_regionRegion));
  result.addRegion(std::move(else_regionRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IfRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "then";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getThenRegion());
  _odsPrinter << ' ' << "else";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getElseRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getCond().getType()), getOuts().getTypes());
}

StringRef IfRegionOp::getDefaultDialect() { return "tfg"; }
void IfRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp IfRegionOp::getThenYield() {
  return cast<YieldOp>(getThenBlock().getTerminator());
}
YieldOp IfRegionOp::getElseYield() {
  return cast<YieldOp>(getElseBlock().getTerminator());
}

void IfRegionOp::getSuccessorRegions(
     ::mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpSuccessorRegions(*this, point, regions);
}
void IfRegionOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpEntrySuccessorRegions(*this, operands, regions);
}
void IfRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
}

LogicalResult IfRegionOp::verify() { return VerifyIfLikeRegionOp(*this); }

BlockArgument IfRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument IfRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument IfRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument IfRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::IfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ReturnOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReturnOpGenericAdaptorBase::ReturnOpGenericAdaptorBase(ReturnOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ReturnOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ArrayAttr ReturnOpGenericAdaptorBase::getControlRetAttrs() {
  auto attr = getControlRetAttrsAttr();
  return attr;
}

} // namespace detail
ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp op) : ReturnOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_control_ret_attrs = getProperties().control_ret_attrs; (void)tblgen_control_ret_attrs;
  if (!tblgen_control_ret_attrs) return emitError(loc, "'tfg.return' op ""requires attribute 'control_ret_attrs'");

  if (tblgen_control_ret_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_control_ret_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_control_ret_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'tfg.return' op ""attribute 'control_ret_attrs' failed to satisfy constraint: an array of dictionaries");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult ReturnOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.control_ret_attrs;
       auto attr = dict.get("control_ret_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `control_ret_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReturnOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.control_ret_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("control_ret_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ReturnOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.control_ret_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReturnOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "control_ret_attrs")
      return prop.control_ret_attrs;
  return std::nullopt;
}

void ReturnOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "control_ret_attrs") {
       prop.control_ret_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.control_ret_attrs)>>(value);
       return;
    }
}

void ReturnOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.control_ret_attrs) attrs.append("control_ret_attrs", prop.control_ret_attrs);
}

::mlir::LogicalResult ReturnOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getControlRetAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops3(attr, "control_ret_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.control_ret_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.control_ret_attrs);
}

::mlir::ArrayAttr ReturnOp::getControlRetAttrs() {
  auto attr = getControlRetAttrsAttr();
  return attr;
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange odsArg_0, ::mlir::ArrayAttr control_ret_attrs) {
  odsState.addOperands(odsArg_0);
  odsState.getOrAddProperties<Properties>().control_ret_attrs = control_ret_attrs;
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange odsArg_0, ::mlir::ArrayAttr control_ret_attrs) {
  odsState.addOperands(odsArg_0);
  odsState.getOrAddProperties<Properties>().control_ret_attrs = control_ret_attrs;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ReturnOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult ReturnOp::verifyInvariantsImpl() {
  auto tblgen_control_ret_attrs = getProperties().control_ret_attrs; (void)tblgen_control_ret_attrs;
  if (!tblgen_control_ret_attrs) return emitOpError("requires attribute 'control_ret_attrs'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_control_ret_attrs, "control_ret_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ReturnOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulCaseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatefulCaseOpGenericAdaptorBase::StatefulCaseOpGenericAdaptorBase(StatefulCaseOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatefulCaseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ArrayAttr StatefulCaseOpGenericAdaptorBase::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseOpGenericAdaptorBase::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseOpGenericAdaptorBase::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatefulCaseOpAdaptor::StatefulCaseOpAdaptor(StatefulCaseOp op) : StatefulCaseOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatefulCaseOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_branches = getProperties().branches; (void)tblgen_branches;
  if (!tblgen_branches) return emitError(loc, "'tfg.StatefulCase' op ""requires attribute 'branches'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;

  if (tblgen_branches && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_branches))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_branches), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::FuncAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'branches' failed to satisfy constraint: an array of functions");

  if (tblgen_Tin && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tin))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tin), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tout))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tout), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatefulCaseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatefulCaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatefulCaseOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulCaseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatefulCaseOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.Tin;
       auto attr = dict.get("Tin");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tin` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tout;
       auto attr = dict.get("Tout");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tout` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.branches;
       auto attr = dict.get("branches");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `branches` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatefulCaseOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.Tin;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tin",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tout;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tout",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.branches;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("branches",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatefulCaseOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.Tin.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tout.getAsOpaquePointer()), 
    llvm::hash_value(prop.branches.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatefulCaseOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "Tin")
      return prop.Tin;

    if (name == "Tout")
      return prop.Tout;

    if (name == "branches")
      return prop.branches;

    if (name == "output_shapes")
      return prop.output_shapes;
  return std::nullopt;
}

void StatefulCaseOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "Tin") {
       prop.Tin = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tin)>>(value);
       return;
    }

    if (name == "Tout") {
       prop.Tout = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tout)>>(value);
       return;
    }

    if (name == "branches") {
       prop.branches = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.branches)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }
}

void StatefulCaseOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.Tin) attrs.append("Tin", prop.Tin);

    if (prop.Tout) attrs.append("Tout", prop.Tout);

    if (prop.branches) attrs.append("branches", prop.branches);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);
}

::mlir::LogicalResult StatefulCaseOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTinAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tin", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getToutAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tout", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBranchesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops0(attr, "branches", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatefulCaseOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.Tin)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tout)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.branches)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulCaseOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.Tin);

  writer.writeOptionalAttribute(prop.Tout);
  writer.writeAttribute(prop.branches);

  writer.writeOptionalAttribute(prop.output_shapes);
}

::mlir::ArrayAttr StatefulCaseOp::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatefulCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches) {
      build(odsBuilder, odsState, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatefulCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().branches = branches;
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().branches = branches;
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulCaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatefulCaseOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatefulCaseOp::verifyInvariantsImpl() {
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_branches = getProperties().branches; (void)tblgen_branches;
  if (!tblgen_branches) return emitOpError("requires attribute 'branches'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops0(*this, tblgen_branches, "branches")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulCaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatefulCaseOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyCaseLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulCaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulCaseRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatefulCaseRegionOpGenericAdaptorBase::StatefulCaseRegionOpGenericAdaptorBase(StatefulCaseRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatefulCaseRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseRegionOpGenericAdaptorBase::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseRegionOpGenericAdaptorBase::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatefulCaseRegionOpAdaptor::StatefulCaseRegionOpAdaptor(StatefulCaseRegionOp op) : StatefulCaseRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatefulCaseRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_branch_attrs = getProperties().branch_attrs; (void)tblgen_branch_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (tblgen_branch_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_branch_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_branch_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatefulCaseRegion' op ""attribute 'branch_attrs' failed to satisfy constraint: an array of dictionaries");

  if (tblgen_region_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_region_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_region_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tfg::RegionAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatefulCaseRegion' op ""attribute 'region_attrs' failed to satisfy constraint: case region preserved attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatefulCaseRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatefulCaseRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulCaseRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatefulCaseRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.branch_attrs;
       auto attr = dict.get("branch_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `branch_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.region_attrs;
       auto attr = dict.get("region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatefulCaseRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.branch_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("branch_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("region_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatefulCaseRegionOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.branch_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.region_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatefulCaseRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "branch_attrs")
      return prop.branch_attrs;

    if (name == "region_attrs")
      return prop.region_attrs;
  return std::nullopt;
}

void StatefulCaseRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "branch_attrs") {
       prop.branch_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.branch_attrs)>>(value);
       return;
    }

    if (name == "region_attrs") {
       prop.region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.region_attrs)>>(value);
       return;
    }
}

void StatefulCaseRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.branch_attrs) attrs.append("branch_attrs", prop.branch_attrs);

    if (prop.region_attrs) attrs.append("region_attrs", prop.region_attrs);
}

::mlir::LogicalResult StatefulCaseRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBranchAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops3(attr, "branch_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops4(attr, "region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatefulCaseRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.branch_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.region_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulCaseRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.branch_attrs);

  writer.writeOptionalAttribute(prop.region_attrs);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseRegionOp::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulCaseRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatefulCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.getOrAddProperties<Properties>().branch_attrs = branch_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.getOrAddProperties<Properties>().branch_attrs = branch_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulCaseRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatefulCaseRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatefulCaseRegionOp::verifyInvariantsImpl() {
  auto tblgen_branch_attrs = getProperties().branch_attrs; (void)tblgen_branch_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_branch_attrs, "branch_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops4(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getBranches())
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "branches", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulCaseRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatefulCaseRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand branch_indexRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> branch_indexOperands(&branch_indexRawOperand, 1);  ::llvm::SMLoc branch_indexOperandsLoc;
  (void)branch_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> branchesRegions;
  ::llvm::ArrayRef<::mlir::Type> branch_indexTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  branch_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(branch_indexRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      branchesRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        branchesRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : branchesRegions)
    if (region->empty()) region->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType branch_index__outs_functionType;
  if (parser.parseType(branch_index__outs_functionType))
    return ::mlir::failure();
  branch_indexTypes = branch_index__outs_functionType.getInputs();
  outsTypes = branch_index__outs_functionType.getResults();
  result.addRegions(branchesRegions);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(branch_indexOperands, branch_indexTypes, branch_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulCaseRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBranchIndex();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
    llvm::interleaveComma(getBranches(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getBranchIndex().getType()), getOuts().getTypes());
}

StringRef StatefulCaseRegionOp::getDefaultDialect() { return "tfg"; }
void StatefulCaseRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatefulCaseRegionOp::branch_yield(unsigned idx) {
  return cast<YieldOp>(getBranchBlock(idx).getTerminator());
}

void StatefulCaseRegionOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpSuccessorRegions(*this, point, regions);
}
void StatefulCaseRegionOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpEntrySuccessorRegions(*this, operands, regions);
}
void StatefulCaseRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
}

LogicalResult StatefulCaseRegionOp::verify() { return VerifyCaseLikeRegionOp(*this); }

BlockArgument StatefulCaseRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatefulCaseRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatefulCaseRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatefulCaseRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulCaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulIfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatefulIfOpGenericAdaptorBase::StatefulIfOpGenericAdaptorBase(StatefulIfOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatefulIfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr StatefulIfOpGenericAdaptorBase::getThenBranch() {
  auto attr = getThenBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatefulIfOpGenericAdaptorBase::getElseBranch() {
  auto attr = getElseBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional<::mlir::Type> StatefulIfOpGenericAdaptorBase::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::std::optional<::mlir::Type>(::llvm::cast<::mlir::Type>(attr.getValue())) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulIfOpGenericAdaptorBase::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulIfOpGenericAdaptorBase::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulIfOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatefulIfOpAdaptor::StatefulIfOpAdaptor(StatefulIfOp op) : StatefulIfOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatefulIfOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_Tcond = getProperties().Tcond; (void)tblgen_Tcond;
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_else_branch = getProperties().else_branch; (void)tblgen_else_branch;
  if (!tblgen_else_branch) return emitError(loc, "'tfg.StatefulIf' op ""requires attribute 'else_branch'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_then_branch = getProperties().then_branch; (void)tblgen_then_branch;
  if (!tblgen_then_branch) return emitError(loc, "'tfg.StatefulIf' op ""requires attribute 'then_branch'");

  if (tblgen_then_branch && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_then_branch))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'then_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_else_branch && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_else_branch))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'else_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_Tcond && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_Tcond))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_Tcond).getValue()))) && ((true))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'Tcond' failed to satisfy constraint: any type attribute");

  if (tblgen_Tin && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tin))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tin), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tout))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tout), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatefulIfOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatefulIfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatefulIfOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulIfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatefulIfOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.Tcond;
       auto attr = dict.get("Tcond");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tcond` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tin;
       auto attr = dict.get("Tin");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tin` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tout;
       auto attr = dict.get("Tout");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tout` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.else_branch;
       auto attr = dict.get("else_branch");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_branch` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_branch;
       auto attr = dict.get("then_branch");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_branch` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatefulIfOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.Tcond;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tcond",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tin;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tin",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tout;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tout",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.else_branch;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_branch",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_branch;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_branch",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatefulIfOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.Tcond.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tin.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tout.getAsOpaquePointer()), 
    llvm::hash_value(prop.else_branch.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_branch.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatefulIfOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "Tcond")
      return prop.Tcond;

    if (name == "Tin")
      return prop.Tin;

    if (name == "Tout")
      return prop.Tout;

    if (name == "else_branch")
      return prop.else_branch;

    if (name == "output_shapes")
      return prop.output_shapes;

    if (name == "then_branch")
      return prop.then_branch;
  return std::nullopt;
}

void StatefulIfOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "Tcond") {
       prop.Tcond = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tcond)>>(value);
       return;
    }

    if (name == "Tin") {
       prop.Tin = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tin)>>(value);
       return;
    }

    if (name == "Tout") {
       prop.Tout = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tout)>>(value);
       return;
    }

    if (name == "else_branch") {
       prop.else_branch = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_branch)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }

    if (name == "then_branch") {
       prop.then_branch = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_branch)>>(value);
       return;
    }
}

void StatefulIfOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.Tcond) attrs.append("Tcond", prop.Tcond);

    if (prop.Tin) attrs.append("Tin", prop.Tin);

    if (prop.Tout) attrs.append("Tout", prop.Tout);

    if (prop.else_branch) attrs.append("else_branch", prop.else_branch);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);

    if (prop.then_branch) attrs.append("then_branch", prop.then_branch);
}

::mlir::LogicalResult StatefulIfOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTcondAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops17(attr, "Tcond", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTinAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tin", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getToutAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tout", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getElseBranchAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "else_branch", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenBranchAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "then_branch", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatefulIfOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.Tcond)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tin)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tout)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.else_branch)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.then_branch)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulIfOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.Tcond);

  writer.writeOptionalAttribute(prop.Tin);

  writer.writeOptionalAttribute(prop.Tout);
  writer.writeAttribute(prop.else_branch);

  writer.writeOptionalAttribute(prop.output_shapes);
  writer.writeAttribute(prop.then_branch);
}

::mlir::tf_type::FuncAttr StatefulIfOp::getThenBranch() {
  auto attr = getThenBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatefulIfOp::getElseBranch() {
  auto attr = getElseBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional<::mlir::Type> StatefulIfOp::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::std::optional<::mlir::Type>(::llvm::cast<::mlir::Type>(attr.getValue())) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulIfOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulIfOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulIfOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatefulIfOp::setTcond(::std::optional<::mlir::Type> attrValue) {
    auto &odsProp = getProperties().Tcond;
    if (attrValue)
      odsProp = ::mlir::TypeAttr::get(*attrValue);
    else
      odsProp = nullptr;
}

void StatefulIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch) {
      build(odsBuilder, odsState, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatefulIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().then_branch = then_branch;
  odsState.getOrAddProperties<Properties>().else_branch = else_branch;
  if (Tcond) {
    odsState.getOrAddProperties<Properties>().Tcond = Tcond;
  }
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().then_branch = then_branch;
  odsState.getOrAddProperties<Properties>().else_branch = else_branch;
  if (Tcond) {
    odsState.getOrAddProperties<Properties>().Tcond = Tcond;
  }
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulIfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatefulIfOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatefulIfOp::verifyInvariantsImpl() {
  auto tblgen_Tcond = getProperties().Tcond; (void)tblgen_Tcond;
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_else_branch = getProperties().else_branch; (void)tblgen_else_branch;
  if (!tblgen_else_branch) return emitOpError("requires attribute 'else_branch'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_then_branch = getProperties().then_branch; (void)tblgen_then_branch;
  if (!tblgen_then_branch) return emitOpError("requires attribute 'then_branch'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_then_branch, "then_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_else_branch, "else_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops17(*this, tblgen_Tcond, "Tcond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulIfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatefulIfOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyIfLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulIfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulIfRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatefulIfRegionOpGenericAdaptorBase::StatefulIfRegionOpGenericAdaptorBase(StatefulIfRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatefulIfRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::std::optional< ::mlir::DictionaryAttr > StatefulIfRegionOpGenericAdaptorBase::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatefulIfRegionOpGenericAdaptorBase::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulIfRegionOpGenericAdaptorBase::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulIfRegionOpGenericAdaptorBase::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
StatefulIfRegionOpAdaptor::StatefulIfRegionOpAdaptor(StatefulIfRegionOp op) : StatefulIfRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatefulIfRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_else_attrs = getProperties().else_attrs; (void)tblgen_else_attrs;
  auto tblgen_else_region_attrs = getProperties().else_region_attrs; (void)tblgen_else_region_attrs;
  auto tblgen_then_attrs = getProperties().then_attrs; (void)tblgen_then_attrs;
  auto tblgen_then_region_attrs = getProperties().then_region_attrs; (void)tblgen_then_region_attrs;

  if (tblgen_then_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_then_attrs))))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'then_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_else_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_else_attrs))))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'else_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_then_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_then_region_attrs))))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'then_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_else_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_else_region_attrs))))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'else_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatefulIfRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatefulIfRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulIfRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatefulIfRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.else_attrs;
       auto attr = dict.get("else_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.else_region_attrs;
       auto attr = dict.get("else_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_attrs;
       auto attr = dict.get("then_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_region_attrs;
       auto attr = dict.get("then_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatefulIfRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.else_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.else_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_region_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatefulIfRegionOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.else_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.else_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_region_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatefulIfRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "else_attrs")
      return prop.else_attrs;

    if (name == "else_region_attrs")
      return prop.else_region_attrs;

    if (name == "then_attrs")
      return prop.then_attrs;

    if (name == "then_region_attrs")
      return prop.then_region_attrs;
  return std::nullopt;
}

void StatefulIfRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "else_attrs") {
       prop.else_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_attrs)>>(value);
       return;
    }

    if (name == "else_region_attrs") {
       prop.else_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_region_attrs)>>(value);
       return;
    }

    if (name == "then_attrs") {
       prop.then_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_attrs)>>(value);
       return;
    }

    if (name == "then_region_attrs") {
       prop.then_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_region_attrs)>>(value);
       return;
    }
}

void StatefulIfRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.else_attrs) attrs.append("else_attrs", prop.else_attrs);

    if (prop.else_region_attrs) attrs.append("else_region_attrs", prop.else_region_attrs);

    if (prop.then_attrs) attrs.append("then_attrs", prop.then_attrs);

    if (prop.then_region_attrs) attrs.append("then_region_attrs", prop.then_region_attrs);
}

::mlir::LogicalResult StatefulIfRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getElseAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "else_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getElseRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "else_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "then_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "then_region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatefulIfRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.else_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.else_region_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.then_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.then_region_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulIfRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.else_attrs);

  writer.writeOptionalAttribute(prop.else_region_attrs);

  writer.writeOptionalAttribute(prop.then_attrs);

  writer.writeOptionalAttribute(prop.then_region_attrs);
}

::std::optional< ::mlir::DictionaryAttr > StatefulIfRegionOp::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatefulIfRegionOp::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulIfRegionOp::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulIfRegionOp::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void StatefulIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.getOrAddProperties<Properties>().then_attrs = then_attrs;
  }
  if (else_attrs) {
    odsState.getOrAddProperties<Properties>().else_attrs = else_attrs;
  }
  if (then_region_attrs) {
    odsState.getOrAddProperties<Properties>().then_region_attrs = then_region_attrs;
  }
  if (else_region_attrs) {
    odsState.getOrAddProperties<Properties>().else_region_attrs = else_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.getOrAddProperties<Properties>().then_attrs = then_attrs;
  }
  if (else_attrs) {
    odsState.getOrAddProperties<Properties>().else_attrs = else_attrs;
  }
  if (then_region_attrs) {
    odsState.getOrAddProperties<Properties>().then_region_attrs = then_region_attrs;
  }
  if (else_region_attrs) {
    odsState.getOrAddProperties<Properties>().else_region_attrs = else_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulIfRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatefulIfRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatefulIfRegionOp::verifyInvariantsImpl() {
  auto tblgen_else_attrs = getProperties().else_attrs; (void)tblgen_else_attrs;
  auto tblgen_else_region_attrs = getProperties().else_region_attrs; (void)tblgen_else_region_attrs;
  auto tblgen_then_attrs = getProperties().then_attrs; (void)tblgen_then_attrs;
  auto tblgen_then_region_attrs = getProperties().then_region_attrs; (void)tblgen_then_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_then_attrs, "then_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_else_attrs, "else_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_then_region_attrs, "then_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_else_region_attrs, "else_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "then_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "else_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulIfRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatefulIfRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(&condRawOperand, 1);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> then_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> else_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> condTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("then"))
    return ::mlir::failure();

  if (parser.parseRegion(*then_regionRegion))
    return ::mlir::failure();

  if (then_regionRegion->empty()) then_regionRegion->emplaceBlock();
  if (parser.parseKeyword("else"))
    return ::mlir::failure();

  if (parser.parseRegion(*else_regionRegion))
    return ::mlir::failure();

  if (else_regionRegion->empty()) else_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType cond__outs_functionType;
  if (parser.parseType(cond__outs_functionType))
    return ::mlir::failure();
  condTypes = cond__outs_functionType.getInputs();
  outsTypes = cond__outs_functionType.getResults();
  result.addRegion(std::move(then_regionRegion));
  result.addRegion(std::move(else_regionRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulIfRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "then";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getThenRegion());
  _odsPrinter << ' ' << "else";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getElseRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getCond().getType()), getOuts().getTypes());
}

StringRef StatefulIfRegionOp::getDefaultDialect() { return "tfg"; }
void StatefulIfRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatefulIfRegionOp::getThenYield() {
  return cast<YieldOp>(getThenBlock().getTerminator());
}
YieldOp StatefulIfRegionOp::getElseYield() {
  return cast<YieldOp>(getElseBlock().getTerminator());
}

void StatefulIfRegionOp::getSuccessorRegions(
     ::mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpSuccessorRegions(*this, point, regions);
}
void StatefulIfRegionOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpEntrySuccessorRegions(*this, operands, regions);
}
void StatefulIfRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
}

LogicalResult StatefulIfRegionOp::verify() { return VerifyIfLikeRegionOp(*this); }

BlockArgument StatefulIfRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatefulIfRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatefulIfRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatefulIfRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulIfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulWhileOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatefulWhileOpGenericAdaptorBase::StatefulWhileOpGenericAdaptorBase(StatefulWhileOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatefulWhileOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr StatefulWhileOpGenericAdaptorBase::getCond() {
  auto attr = getCondAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatefulWhileOpGenericAdaptorBase::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

uint64_t StatefulWhileOpGenericAdaptorBase::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > StatefulWhileOpGenericAdaptorBase::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulWhileOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatefulWhileOpAdaptor::StatefulWhileOpAdaptor(StatefulWhileOp op) : StatefulWhileOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatefulWhileOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitError(loc, "'tfg.StatefulWhile' op ""requires attribute 'body'");
  auto tblgen_cond = getProperties().cond; (void)tblgen_cond;
  if (!tblgen_cond) return emitError(loc, "'tfg.StatefulWhile' op ""requires attribute 'cond'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitError(loc, "'tfg.StatefulWhile' op ""requires attribute 'parallel_iterations'");

  if (tblgen_cond && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_cond))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'cond' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_body && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_body))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_parallel_iterations && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_parallel_iterations))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_parallel_iterations).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_T && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_T))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_T), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'T' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatefulWhileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatefulWhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatefulWhileOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulWhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatefulWhileOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.T;
       auto attr = dict.get("T");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `T` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body;
       auto attr = dict.get("body");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond;
       auto attr = dict.get("cond");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.parallel_iterations;
       auto attr = dict.get("parallel_iterations");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `parallel_iterations` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatefulWhileOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.T;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("T",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.parallel_iterations;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("parallel_iterations",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatefulWhileOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.T.getAsOpaquePointer()), 
    llvm::hash_value(prop.body.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()), 
    llvm::hash_value(prop.parallel_iterations.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatefulWhileOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "T")
      return prop.T;

    if (name == "body")
      return prop.body;

    if (name == "cond")
      return prop.cond;

    if (name == "output_shapes")
      return prop.output_shapes;

    if (name == "parallel_iterations")
      return prop.parallel_iterations;
  return std::nullopt;
}

void StatefulWhileOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "T") {
       prop.T = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.T)>>(value);
       return;
    }

    if (name == "body") {
       prop.body = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body)>>(value);
       return;
    }

    if (name == "cond") {
       prop.cond = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }

    if (name == "parallel_iterations") {
       prop.parallel_iterations = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.parallel_iterations)>>(value);
       return;
    }
}

void StatefulWhileOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.T) attrs.append("T", prop.T);

    if (prop.body) attrs.append("body", prop.body);

    if (prop.cond) attrs.append("cond", prop.cond);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);

    if (prop.parallel_iterations) attrs.append("parallel_iterations", prop.parallel_iterations);
}

::mlir::LogicalResult StatefulWhileOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "T", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "body", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "cond", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getParallelIterationsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops18(attr, "parallel_iterations", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatefulWhileOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.T)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.body)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.cond)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.parallel_iterations)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulWhileOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.T);
  writer.writeAttribute(prop.body);
  writer.writeAttribute(prop.cond);

  writer.writeOptionalAttribute(prop.output_shapes);
  writer.writeAttribute(prop.parallel_iterations);
}

::mlir::tf_type::FuncAttr StatefulWhileOp::getCond() {
  auto attr = getCondAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatefulWhileOp::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

uint64_t StatefulWhileOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > StatefulWhileOp::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatefulWhileOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatefulWhileOp::setParallelIterations(uint64_t attrValue) {
  getProperties().parallel_iterations = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations) {
      build(odsBuilder, odsState, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatefulWhileOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatefulWhileOp::verifyInvariantsImpl() {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitOpError("requires attribute 'body'");
  auto tblgen_cond = getProperties().cond; (void)tblgen_cond;
  if (!tblgen_cond) return emitOpError("requires attribute 'cond'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitOpError("requires attribute 'parallel_iterations'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_cond, "cond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulWhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatefulWhileOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyWhileLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulWhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulWhileRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatefulWhileRegionOpGenericAdaptorBase::StatefulWhileRegionOpGenericAdaptorBase(StatefulWhileRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatefulWhileRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

uint64_t StatefulWhileRegionOpGenericAdaptorBase::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > StatefulWhileRegionOpGenericAdaptorBase::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatefulWhileRegionOpGenericAdaptorBase::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOpGenericAdaptorBase::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOpGenericAdaptorBase::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
StatefulWhileRegionOpAdaptor::StatefulWhileRegionOpAdaptor(StatefulWhileRegionOp op) : StatefulWhileRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatefulWhileRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_body_region_attrs = getProperties().body_region_attrs; (void)tblgen_body_region_attrs;
  auto tblgen_cond_attrs = getProperties().cond_attrs; (void)tblgen_cond_attrs;
  auto tblgen_cond_region_attrs = getProperties().cond_region_attrs; (void)tblgen_cond_region_attrs;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitError(loc, "'tfg.StatefulWhileRegion' op ""requires attribute 'parallel_iterations'");

  if (tblgen_parallel_iterations && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_parallel_iterations))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_parallel_iterations).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_cond_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_cond_attrs))))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'cond_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_body_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_body_attrs))))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_cond_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_cond_region_attrs))))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'cond_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_body_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_body_region_attrs))))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'body_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatefulWhileRegionOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange StatefulWhileRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange StatefulWhileRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulWhileRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatefulWhileRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.body_attrs;
       auto attr = dict.get("body_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body_region_attrs;
       auto attr = dict.get("body_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond_attrs;
       auto attr = dict.get("cond_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond_region_attrs;
       auto attr = dict.get("cond_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.parallel_iterations;
       auto attr = dict.get("parallel_iterations");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `parallel_iterations` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute StatefulWhileRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.body_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.parallel_iterations;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("parallel_iterations",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes",
                                              ::mlir::DenseI32ArrayAttr::get(ctx, propStorage)));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatefulWhileRegionOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.body_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.body_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.parallel_iterations.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> StatefulWhileRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "body_attrs")
      return prop.body_attrs;

    if (name == "body_region_attrs")
      return prop.body_region_attrs;

    if (name == "cond_attrs")
      return prop.cond_attrs;

    if (name == "cond_region_attrs")
      return prop.cond_region_attrs;

    if (name == "parallel_iterations")
      return prop.parallel_iterations;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes);
  return std::nullopt;
}

void StatefulWhileRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "body_attrs") {
       prop.body_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_attrs)>>(value);
       return;
    }

    if (name == "body_region_attrs") {
       prop.body_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_region_attrs)>>(value);
       return;
    }

    if (name == "cond_attrs") {
       prop.cond_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond_attrs)>>(value);
       return;
    }

    if (name == "cond_region_attrs") {
       prop.cond_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond_region_attrs)>>(value);
       return;
    }

    if (name == "parallel_iterations") {
       prop.parallel_iterations = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.parallel_iterations)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void StatefulWhileRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.body_attrs) attrs.append("body_attrs", prop.body_attrs);

    if (prop.body_region_attrs) attrs.append("body_region_attrs", prop.body_region_attrs);

    if (prop.cond_attrs) attrs.append("cond_attrs", prop.cond_attrs);

    if (prop.cond_region_attrs) attrs.append("cond_region_attrs", prop.cond_region_attrs);

    if (prop.parallel_iterations) attrs.append("parallel_iterations", prop.parallel_iterations);
  attrs.append("operandSegmentSizes", ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes));
}

::mlir::LogicalResult StatefulWhileRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "body_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "body_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "cond_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "cond_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getParallelIterationsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops18(attr, "parallel_iterations", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatefulWhileRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.body_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.body_region_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cond_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cond_region_attrs)))
    return ::mlir::failure();

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  if (::mlir::failed(reader.readAttribute(prop.parallel_iterations)))
    return ::mlir::failure();

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void StatefulWhileRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.body_attrs);

  writer.writeOptionalAttribute(prop.body_region_attrs);

  writer.writeOptionalAttribute(prop.cond_attrs);

  writer.writeOptionalAttribute(prop.cond_region_attrs);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}
  writer.writeAttribute(prop.parallel_iterations);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

uint64_t StatefulWhileRegionOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > StatefulWhileRegionOp::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatefulWhileRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOp::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOp::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void StatefulWhileRegionOp::setParallelIterations(uint64_t attrValue) {
  getProperties().parallel_iterations = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatefulWhileRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatefulWhileRegionOp::verifyInvariantsImpl() {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_body_region_attrs = getProperties().body_region_attrs; (void)tblgen_body_region_attrs;
  auto tblgen_cond_attrs = getProperties().cond_attrs; (void)tblgen_cond_attrs;
  auto tblgen_cond_region_attrs = getProperties().cond_region_attrs; (void)tblgen_cond_region_attrs;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitOpError("requires attribute 'parallel_iterations'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_cond_attrs, "cond_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_cond_region_attrs, "cond_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_body_region_attrs, "body_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "cond_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulWhileRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatefulWhileRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> cond_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> initTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*cond_regionRegion))
    return ::mlir::failure();

  if (cond_regionRegion->empty()) cond_regionRegion->emplaceBlock();
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  ::mlir::FunctionType init__outs_functionType;
  if (parser.parseType(init__outs_functionType))
    return ::mlir::failure();
  initTypes = init__outs_functionType.getInputs();
  outsTypes = init__outs_functionType.getResults();
  }
  result.addRegion(std::move(cond_regionRegion));
  result.addRegion(std::move(body_regionRegion));
::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}), result.getOrAddProperties<StatefulWhileRegionOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulWhileRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getCondRegion());
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getInit().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printFunctionalType(getInit().getTypes(), getOuts().getTypes());
  }
}

StringRef StatefulWhileRegionOp::getDefaultDialect() { return "tfg"; }
void StatefulWhileRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

ConditionOp StatefulWhileRegionOp::getCondCondition() {
  return cast<ConditionOp>(getCondBlock().getTerminator());
}
YieldOp StatefulWhileRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}

OperandRange StatefulWhileRegionOp::getEntrySuccessorOperands(
     ::mlir::RegionBranchPoint point) {
  return getInit();
}
void StatefulWhileRegionOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return GetWhileLikeRegionOpSuccessorRegions(*this, point, regions);
}

LogicalResult StatefulWhileRegionOp::verify() { return VerifyWhileLikeRegionOp(*this); }

BlockArgument StatefulWhileRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatefulWhileRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatefulWhileRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatefulWhileRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulWhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessCaseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatelessCaseOpGenericAdaptorBase::StatelessCaseOpGenericAdaptorBase(StatelessCaseOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatelessCaseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ArrayAttr StatelessCaseOpGenericAdaptorBase::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseOpGenericAdaptorBase::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseOpGenericAdaptorBase::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatelessCaseOpAdaptor::StatelessCaseOpAdaptor(StatelessCaseOp op) : StatelessCaseOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatelessCaseOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_branches = getProperties().branches; (void)tblgen_branches;
  if (!tblgen_branches) return emitError(loc, "'tfg.StatelessCase' op ""requires attribute 'branches'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;

  if (tblgen_branches && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_branches))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_branches), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::FuncAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'branches' failed to satisfy constraint: an array of functions");

  if (tblgen_Tin && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tin))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tin), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tout))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tout), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatelessCaseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatelessCaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatelessCaseOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessCaseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatelessCaseOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.Tin;
       auto attr = dict.get("Tin");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tin` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tout;
       auto attr = dict.get("Tout");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tout` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.branches;
       auto attr = dict.get("branches");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `branches` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatelessCaseOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.Tin;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tin",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tout;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tout",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.branches;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("branches",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatelessCaseOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.Tin.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tout.getAsOpaquePointer()), 
    llvm::hash_value(prop.branches.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatelessCaseOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "Tin")
      return prop.Tin;

    if (name == "Tout")
      return prop.Tout;

    if (name == "branches")
      return prop.branches;

    if (name == "output_shapes")
      return prop.output_shapes;
  return std::nullopt;
}

void StatelessCaseOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "Tin") {
       prop.Tin = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tin)>>(value);
       return;
    }

    if (name == "Tout") {
       prop.Tout = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tout)>>(value);
       return;
    }

    if (name == "branches") {
       prop.branches = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.branches)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }
}

void StatelessCaseOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.Tin) attrs.append("Tin", prop.Tin);

    if (prop.Tout) attrs.append("Tout", prop.Tout);

    if (prop.branches) attrs.append("branches", prop.branches);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);
}

::mlir::LogicalResult StatelessCaseOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTinAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tin", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getToutAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tout", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBranchesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops0(attr, "branches", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatelessCaseOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.Tin)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tout)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.branches)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessCaseOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.Tin);

  writer.writeOptionalAttribute(prop.Tout);
  writer.writeAttribute(prop.branches);

  writer.writeOptionalAttribute(prop.output_shapes);
}

::mlir::ArrayAttr StatelessCaseOp::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatelessCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches) {
      build(odsBuilder, odsState, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatelessCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().branches = branches;
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().branches = branches;
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessCaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatelessCaseOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatelessCaseOp::verifyInvariantsImpl() {
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_branches = getProperties().branches; (void)tblgen_branches;
  if (!tblgen_branches) return emitOpError("requires attribute 'branches'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops0(*this, tblgen_branches, "branches")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessCaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatelessCaseOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyCaseLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessCaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessCaseRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatelessCaseRegionOpGenericAdaptorBase::StatelessCaseRegionOpGenericAdaptorBase(StatelessCaseRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatelessCaseRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseRegionOpGenericAdaptorBase::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseRegionOpGenericAdaptorBase::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatelessCaseRegionOpAdaptor::StatelessCaseRegionOpAdaptor(StatelessCaseRegionOp op) : StatelessCaseRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatelessCaseRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_branch_attrs = getProperties().branch_attrs; (void)tblgen_branch_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (tblgen_branch_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_branch_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_branch_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatelessCaseRegion' op ""attribute 'branch_attrs' failed to satisfy constraint: an array of dictionaries");

  if (tblgen_region_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_region_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_region_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tfg::RegionAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatelessCaseRegion' op ""attribute 'region_attrs' failed to satisfy constraint: case region preserved attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatelessCaseRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatelessCaseRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessCaseRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatelessCaseRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.branch_attrs;
       auto attr = dict.get("branch_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `branch_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.region_attrs;
       auto attr = dict.get("region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatelessCaseRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.branch_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("branch_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("region_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatelessCaseRegionOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.branch_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.region_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatelessCaseRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "branch_attrs")
      return prop.branch_attrs;

    if (name == "region_attrs")
      return prop.region_attrs;
  return std::nullopt;
}

void StatelessCaseRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "branch_attrs") {
       prop.branch_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.branch_attrs)>>(value);
       return;
    }

    if (name == "region_attrs") {
       prop.region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.region_attrs)>>(value);
       return;
    }
}

void StatelessCaseRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.branch_attrs) attrs.append("branch_attrs", prop.branch_attrs);

    if (prop.region_attrs) attrs.append("region_attrs", prop.region_attrs);
}

::mlir::LogicalResult StatelessCaseRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBranchAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops3(attr, "branch_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops4(attr, "region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatelessCaseRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.branch_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.region_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessCaseRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.branch_attrs);

  writer.writeOptionalAttribute(prop.region_attrs);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseRegionOp::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessCaseRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatelessCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.getOrAddProperties<Properties>().branch_attrs = branch_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.getOrAddProperties<Properties>().branch_attrs = branch_attrs;
  }
  if (region_attrs) {
    odsState.getOrAddProperties<Properties>().region_attrs = region_attrs;
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessCaseRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatelessCaseRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatelessCaseRegionOp::verifyInvariantsImpl() {
  auto tblgen_branch_attrs = getProperties().branch_attrs; (void)tblgen_branch_attrs;
  auto tblgen_region_attrs = getProperties().region_attrs; (void)tblgen_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_branch_attrs, "branch_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops4(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getBranches())
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "branches", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessCaseRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatelessCaseRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand branch_indexRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> branch_indexOperands(&branch_indexRawOperand, 1);  ::llvm::SMLoc branch_indexOperandsLoc;
  (void)branch_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> branchesRegions;
  ::llvm::ArrayRef<::mlir::Type> branch_indexTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  branch_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(branch_indexRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      branchesRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        branchesRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : branchesRegions)
    if (region->empty()) region->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType branch_index__outs_functionType;
  if (parser.parseType(branch_index__outs_functionType))
    return ::mlir::failure();
  branch_indexTypes = branch_index__outs_functionType.getInputs();
  outsTypes = branch_index__outs_functionType.getResults();
  result.addRegions(branchesRegions);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(branch_indexOperands, branch_indexTypes, branch_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessCaseRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBranchIndex();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
    llvm::interleaveComma(getBranches(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getBranchIndex().getType()), getOuts().getTypes());
}

StringRef StatelessCaseRegionOp::getDefaultDialect() { return "tfg"; }
void StatelessCaseRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatelessCaseRegionOp::branch_yield(unsigned idx) {
  return cast<YieldOp>(getBranchBlock(idx).getTerminator());
}

void StatelessCaseRegionOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpSuccessorRegions(*this, point, regions);
}
void StatelessCaseRegionOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpEntrySuccessorRegions(*this, operands, regions);
}
void StatelessCaseRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
}

LogicalResult StatelessCaseRegionOp::verify() { return VerifyCaseLikeRegionOp(*this); }

BlockArgument StatelessCaseRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatelessCaseRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatelessCaseRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatelessCaseRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessCaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessIfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatelessIfOpGenericAdaptorBase::StatelessIfOpGenericAdaptorBase(StatelessIfOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatelessIfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr StatelessIfOpGenericAdaptorBase::getThenBranch() {
  auto attr = getThenBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatelessIfOpGenericAdaptorBase::getElseBranch() {
  auto attr = getElseBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional<::mlir::Type> StatelessIfOpGenericAdaptorBase::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::std::optional<::mlir::Type>(::llvm::cast<::mlir::Type>(attr.getValue())) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessIfOpGenericAdaptorBase::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessIfOpGenericAdaptorBase::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessIfOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatelessIfOpAdaptor::StatelessIfOpAdaptor(StatelessIfOp op) : StatelessIfOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatelessIfOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_Tcond = getProperties().Tcond; (void)tblgen_Tcond;
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_else_branch = getProperties().else_branch; (void)tblgen_else_branch;
  if (!tblgen_else_branch) return emitError(loc, "'tfg.StatelessIf' op ""requires attribute 'else_branch'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_then_branch = getProperties().then_branch; (void)tblgen_then_branch;
  if (!tblgen_then_branch) return emitError(loc, "'tfg.StatelessIf' op ""requires attribute 'then_branch'");

  if (tblgen_then_branch && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_then_branch))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'then_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_else_branch && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_else_branch))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'else_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_Tcond && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_Tcond))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_Tcond).getValue()))) && ((true))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'Tcond' failed to satisfy constraint: any type attribute");

  if (tblgen_Tin && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tin))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tin), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_Tout))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_Tout), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatelessIfOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatelessIfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatelessIfOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessIfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatelessIfOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.Tcond;
       auto attr = dict.get("Tcond");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tcond` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tin;
       auto attr = dict.get("Tin");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tin` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.Tout;
       auto attr = dict.get("Tout");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `Tout` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.else_branch;
       auto attr = dict.get("else_branch");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_branch` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_branch;
       auto attr = dict.get("then_branch");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_branch` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatelessIfOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.Tcond;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tcond",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tin;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tin",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.Tout;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("Tout",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.else_branch;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_branch",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_branch;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_branch",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatelessIfOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.Tcond.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tin.getAsOpaquePointer()), 
    llvm::hash_value(prop.Tout.getAsOpaquePointer()), 
    llvm::hash_value(prop.else_branch.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_branch.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatelessIfOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "Tcond")
      return prop.Tcond;

    if (name == "Tin")
      return prop.Tin;

    if (name == "Tout")
      return prop.Tout;

    if (name == "else_branch")
      return prop.else_branch;

    if (name == "output_shapes")
      return prop.output_shapes;

    if (name == "then_branch")
      return prop.then_branch;
  return std::nullopt;
}

void StatelessIfOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "Tcond") {
       prop.Tcond = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tcond)>>(value);
       return;
    }

    if (name == "Tin") {
       prop.Tin = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tin)>>(value);
       return;
    }

    if (name == "Tout") {
       prop.Tout = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.Tout)>>(value);
       return;
    }

    if (name == "else_branch") {
       prop.else_branch = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_branch)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }

    if (name == "then_branch") {
       prop.then_branch = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_branch)>>(value);
       return;
    }
}

void StatelessIfOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.Tcond) attrs.append("Tcond", prop.Tcond);

    if (prop.Tin) attrs.append("Tin", prop.Tin);

    if (prop.Tout) attrs.append("Tout", prop.Tout);

    if (prop.else_branch) attrs.append("else_branch", prop.else_branch);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);

    if (prop.then_branch) attrs.append("then_branch", prop.then_branch);
}

::mlir::LogicalResult StatelessIfOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTcondAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops17(attr, "Tcond", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTinAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tin", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getToutAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "Tout", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getElseBranchAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "else_branch", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenBranchAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "then_branch", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatelessIfOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.Tcond)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tin)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.Tout)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.else_branch)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.then_branch)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessIfOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.Tcond);

  writer.writeOptionalAttribute(prop.Tin);

  writer.writeOptionalAttribute(prop.Tout);
  writer.writeAttribute(prop.else_branch);

  writer.writeOptionalAttribute(prop.output_shapes);
  writer.writeAttribute(prop.then_branch);
}

::mlir::tf_type::FuncAttr StatelessIfOp::getThenBranch() {
  auto attr = getThenBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatelessIfOp::getElseBranch() {
  auto attr = getElseBranchAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::std::optional<::mlir::Type> StatelessIfOp::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::std::optional<::mlir::Type>(::llvm::cast<::mlir::Type>(attr.getValue())) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessIfOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessIfOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessIfOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatelessIfOp::setTcond(::std::optional<::mlir::Type> attrValue) {
    auto &odsProp = getProperties().Tcond;
    if (attrValue)
      odsProp = ::mlir::TypeAttr::get(*attrValue);
    else
      odsProp = nullptr;
}

void StatelessIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch) {
      build(odsBuilder, odsState, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatelessIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().then_branch = then_branch;
  odsState.getOrAddProperties<Properties>().else_branch = else_branch;
  if (Tcond) {
    odsState.getOrAddProperties<Properties>().Tcond = Tcond;
  }
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().then_branch = then_branch;
  odsState.getOrAddProperties<Properties>().else_branch = else_branch;
  if (Tcond) {
    odsState.getOrAddProperties<Properties>().Tcond = Tcond;
  }
  if (Tin) {
    odsState.getOrAddProperties<Properties>().Tin = Tin;
  }
  if (Tout) {
    odsState.getOrAddProperties<Properties>().Tout = Tout;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessIfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatelessIfOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatelessIfOp::verifyInvariantsImpl() {
  auto tblgen_Tcond = getProperties().Tcond; (void)tblgen_Tcond;
  auto tblgen_Tin = getProperties().Tin; (void)tblgen_Tin;
  auto tblgen_Tout = getProperties().Tout; (void)tblgen_Tout;
  auto tblgen_else_branch = getProperties().else_branch; (void)tblgen_else_branch;
  if (!tblgen_else_branch) return emitOpError("requires attribute 'else_branch'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_then_branch = getProperties().then_branch; (void)tblgen_then_branch;
  if (!tblgen_then_branch) return emitOpError("requires attribute 'then_branch'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_then_branch, "then_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_else_branch, "else_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops17(*this, tblgen_Tcond, "Tcond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessIfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatelessIfOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyIfLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessIfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessIfRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatelessIfRegionOpGenericAdaptorBase::StatelessIfRegionOpGenericAdaptorBase(StatelessIfRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatelessIfRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::std::optional< ::mlir::DictionaryAttr > StatelessIfRegionOpGenericAdaptorBase::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatelessIfRegionOpGenericAdaptorBase::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessIfRegionOpGenericAdaptorBase::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessIfRegionOpGenericAdaptorBase::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
StatelessIfRegionOpAdaptor::StatelessIfRegionOpAdaptor(StatelessIfRegionOp op) : StatelessIfRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatelessIfRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_else_attrs = getProperties().else_attrs; (void)tblgen_else_attrs;
  auto tblgen_else_region_attrs = getProperties().else_region_attrs; (void)tblgen_else_region_attrs;
  auto tblgen_then_attrs = getProperties().then_attrs; (void)tblgen_then_attrs;
  auto tblgen_then_region_attrs = getProperties().then_region_attrs; (void)tblgen_then_region_attrs;

  if (tblgen_then_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_then_attrs))))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'then_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_else_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_else_attrs))))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'else_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_then_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_then_region_attrs))))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'then_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_else_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_else_region_attrs))))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'else_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatelessIfRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatelessIfRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessIfRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatelessIfRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.else_attrs;
       auto attr = dict.get("else_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.else_region_attrs;
       auto attr = dict.get("else_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `else_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_attrs;
       auto attr = dict.get("then_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.then_region_attrs;
       auto attr = dict.get("then_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `then_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatelessIfRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.else_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.else_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("else_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.then_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("then_region_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatelessIfRegionOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.else_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.else_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.then_region_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatelessIfRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "else_attrs")
      return prop.else_attrs;

    if (name == "else_region_attrs")
      return prop.else_region_attrs;

    if (name == "then_attrs")
      return prop.then_attrs;

    if (name == "then_region_attrs")
      return prop.then_region_attrs;
  return std::nullopt;
}

void StatelessIfRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "else_attrs") {
       prop.else_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_attrs)>>(value);
       return;
    }

    if (name == "else_region_attrs") {
       prop.else_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.else_region_attrs)>>(value);
       return;
    }

    if (name == "then_attrs") {
       prop.then_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_attrs)>>(value);
       return;
    }

    if (name == "then_region_attrs") {
       prop.then_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.then_region_attrs)>>(value);
       return;
    }
}

void StatelessIfRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.else_attrs) attrs.append("else_attrs", prop.else_attrs);

    if (prop.else_region_attrs) attrs.append("else_region_attrs", prop.else_region_attrs);

    if (prop.then_attrs) attrs.append("then_attrs", prop.then_attrs);

    if (prop.then_region_attrs) attrs.append("then_region_attrs", prop.then_region_attrs);
}

::mlir::LogicalResult StatelessIfRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getElseAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "else_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getElseRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "else_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "then_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getThenRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "then_region_attrs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatelessIfRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.else_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.else_region_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.then_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.then_region_attrs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessIfRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.else_attrs);

  writer.writeOptionalAttribute(prop.else_region_attrs);

  writer.writeOptionalAttribute(prop.then_attrs);

  writer.writeOptionalAttribute(prop.then_region_attrs);
}

::std::optional< ::mlir::DictionaryAttr > StatelessIfRegionOp::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatelessIfRegionOp::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessIfRegionOp::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessIfRegionOp::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void StatelessIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.getOrAddProperties<Properties>().then_attrs = then_attrs;
  }
  if (else_attrs) {
    odsState.getOrAddProperties<Properties>().else_attrs = else_attrs;
  }
  if (then_region_attrs) {
    odsState.getOrAddProperties<Properties>().then_region_attrs = then_region_attrs;
  }
  if (else_region_attrs) {
    odsState.getOrAddProperties<Properties>().else_region_attrs = else_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.getOrAddProperties<Properties>().then_attrs = then_attrs;
  }
  if (else_attrs) {
    odsState.getOrAddProperties<Properties>().else_attrs = else_attrs;
  }
  if (then_region_attrs) {
    odsState.getOrAddProperties<Properties>().then_region_attrs = then_region_attrs;
  }
  if (else_region_attrs) {
    odsState.getOrAddProperties<Properties>().else_region_attrs = else_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessIfRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatelessIfRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatelessIfRegionOp::verifyInvariantsImpl() {
  auto tblgen_else_attrs = getProperties().else_attrs; (void)tblgen_else_attrs;
  auto tblgen_else_region_attrs = getProperties().else_region_attrs; (void)tblgen_else_region_attrs;
  auto tblgen_then_attrs = getProperties().then_attrs; (void)tblgen_then_attrs;
  auto tblgen_then_region_attrs = getProperties().then_region_attrs; (void)tblgen_then_region_attrs;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_then_attrs, "then_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_else_attrs, "else_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_then_region_attrs, "then_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_else_region_attrs, "else_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "then_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "else_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessIfRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatelessIfRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(&condRawOperand, 1);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> then_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> else_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> condTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("then"))
    return ::mlir::failure();

  if (parser.parseRegion(*then_regionRegion))
    return ::mlir::failure();

  if (then_regionRegion->empty()) then_regionRegion->emplaceBlock();
  if (parser.parseKeyword("else"))
    return ::mlir::failure();

  if (parser.parseRegion(*else_regionRegion))
    return ::mlir::failure();

  if (else_regionRegion->empty()) else_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType cond__outs_functionType;
  if (parser.parseType(cond__outs_functionType))
    return ::mlir::failure();
  condTypes = cond__outs_functionType.getInputs();
  outsTypes = cond__outs_functionType.getResults();
  result.addRegion(std::move(then_regionRegion));
  result.addRegion(std::move(else_regionRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessIfRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "then";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getThenRegion());
  _odsPrinter << ' ' << "else";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getElseRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getCond().getType()), getOuts().getTypes());
}

StringRef StatelessIfRegionOp::getDefaultDialect() { return "tfg"; }
void StatelessIfRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatelessIfRegionOp::getThenYield() {
  return cast<YieldOp>(getThenBlock().getTerminator());
}
YieldOp StatelessIfRegionOp::getElseYield() {
  return cast<YieldOp>(getElseBlock().getTerminator());
}

void StatelessIfRegionOp::getSuccessorRegions(
     ::mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpSuccessorRegions(*this, point, regions);
}
void StatelessIfRegionOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpEntrySuccessorRegions(*this, operands, regions);
}
void StatelessIfRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
}

LogicalResult StatelessIfRegionOp::verify() { return VerifyIfLikeRegionOp(*this); }

BlockArgument StatelessIfRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatelessIfRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatelessIfRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatelessIfRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessIfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessWhileOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatelessWhileOpGenericAdaptorBase::StatelessWhileOpGenericAdaptorBase(StatelessWhileOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatelessWhileOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr StatelessWhileOpGenericAdaptorBase::getCond() {
  auto attr = getCondAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatelessWhileOpGenericAdaptorBase::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

uint64_t StatelessWhileOpGenericAdaptorBase::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > StatelessWhileOpGenericAdaptorBase::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessWhileOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
StatelessWhileOpAdaptor::StatelessWhileOpAdaptor(StatelessWhileOp op) : StatelessWhileOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatelessWhileOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitError(loc, "'tfg.StatelessWhile' op ""requires attribute 'body'");
  auto tblgen_cond = getProperties().cond; (void)tblgen_cond;
  if (!tblgen_cond) return emitError(loc, "'tfg.StatelessWhile' op ""requires attribute 'cond'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitError(loc, "'tfg.StatelessWhile' op ""requires attribute 'parallel_iterations'");

  if (tblgen_cond && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_cond))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'cond' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_body && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_body))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_parallel_iterations && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_parallel_iterations))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_parallel_iterations).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_T && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_T))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_T), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'T' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatelessWhileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatelessWhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange StatelessWhileOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessWhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatelessWhileOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.T;
       auto attr = dict.get("T");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `T` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body;
       auto attr = dict.get("body");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond;
       auto attr = dict.get("cond");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.parallel_iterations;
       auto attr = dict.get("parallel_iterations");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `parallel_iterations` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute StatelessWhileOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.T;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("T",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.parallel_iterations;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("parallel_iterations",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatelessWhileOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.T.getAsOpaquePointer()), 
    llvm::hash_value(prop.body.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()), 
    llvm::hash_value(prop.parallel_iterations.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> StatelessWhileOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "T")
      return prop.T;

    if (name == "body")
      return prop.body;

    if (name == "cond")
      return prop.cond;

    if (name == "output_shapes")
      return prop.output_shapes;

    if (name == "parallel_iterations")
      return prop.parallel_iterations;
  return std::nullopt;
}

void StatelessWhileOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "T") {
       prop.T = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.T)>>(value);
       return;
    }

    if (name == "body") {
       prop.body = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body)>>(value);
       return;
    }

    if (name == "cond") {
       prop.cond = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }

    if (name == "parallel_iterations") {
       prop.parallel_iterations = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.parallel_iterations)>>(value);
       return;
    }
}

void StatelessWhileOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.T) attrs.append("T", prop.T);

    if (prop.body) attrs.append("body", prop.body);

    if (prop.cond) attrs.append("cond", prop.cond);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);

    if (prop.parallel_iterations) attrs.append("parallel_iterations", prop.parallel_iterations);
}

::mlir::LogicalResult StatelessWhileOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "T", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "body", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "cond", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getParallelIterationsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops18(attr, "parallel_iterations", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatelessWhileOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.T)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.body)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.cond)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.parallel_iterations)))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessWhileOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.T);
  writer.writeAttribute(prop.body);
  writer.writeAttribute(prop.cond);

  writer.writeOptionalAttribute(prop.output_shapes);
  writer.writeAttribute(prop.parallel_iterations);
}

::mlir::tf_type::FuncAttr StatelessWhileOp::getCond() {
  auto attr = getCondAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr StatelessWhileOp::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

uint64_t StatelessWhileOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > StatelessWhileOp::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > StatelessWhileOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void StatelessWhileOp::setParallelIterations(uint64_t attrValue) {
  getProperties().parallel_iterations = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations) {
      build(odsBuilder, odsState, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatelessWhileOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatelessWhileOp::verifyInvariantsImpl() {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitOpError("requires attribute 'body'");
  auto tblgen_cond = getProperties().cond; (void)tblgen_cond;
  if (!tblgen_cond) return emitOpError("requires attribute 'cond'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitOpError("requires attribute 'parallel_iterations'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_cond, "cond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessWhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatelessWhileOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyWhileLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessWhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessWhileRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StatelessWhileRegionOpGenericAdaptorBase::StatelessWhileRegionOpGenericAdaptorBase(StatelessWhileRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> StatelessWhileRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

uint64_t StatelessWhileRegionOpGenericAdaptorBase::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > StatelessWhileRegionOpGenericAdaptorBase::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatelessWhileRegionOpGenericAdaptorBase::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOpGenericAdaptorBase::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOpGenericAdaptorBase::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
StatelessWhileRegionOpAdaptor::StatelessWhileRegionOpAdaptor(StatelessWhileRegionOp op) : StatelessWhileRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult StatelessWhileRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_body_region_attrs = getProperties().body_region_attrs; (void)tblgen_body_region_attrs;
  auto tblgen_cond_attrs = getProperties().cond_attrs; (void)tblgen_cond_attrs;
  auto tblgen_cond_region_attrs = getProperties().cond_region_attrs; (void)tblgen_cond_region_attrs;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitError(loc, "'tfg.StatelessWhileRegion' op ""requires attribute 'parallel_iterations'");

  if (tblgen_parallel_iterations && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_parallel_iterations))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_parallel_iterations).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_cond_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_cond_attrs))))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'cond_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_body_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_body_attrs))))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_cond_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_cond_region_attrs))))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'cond_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_body_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_body_region_attrs))))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'body_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatelessWhileRegionOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange StatelessWhileRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange StatelessWhileRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessWhileRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult StatelessWhileRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.body_attrs;
       auto attr = dict.get("body_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body_region_attrs;
       auto attr = dict.get("body_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond_attrs;
       auto attr = dict.get("cond_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond_region_attrs;
       auto attr = dict.get("cond_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.parallel_iterations;
       auto attr = dict.get("parallel_iterations");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `parallel_iterations` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute StatelessWhileRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.body_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.parallel_iterations;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("parallel_iterations",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes",
                                              ::mlir::DenseI32ArrayAttr::get(ctx, propStorage)));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code StatelessWhileRegionOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.body_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.body_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.parallel_iterations.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> StatelessWhileRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "body_attrs")
      return prop.body_attrs;

    if (name == "body_region_attrs")
      return prop.body_region_attrs;

    if (name == "cond_attrs")
      return prop.cond_attrs;

    if (name == "cond_region_attrs")
      return prop.cond_region_attrs;

    if (name == "parallel_iterations")
      return prop.parallel_iterations;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes);
  return std::nullopt;
}

void StatelessWhileRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "body_attrs") {
       prop.body_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_attrs)>>(value);
       return;
    }

    if (name == "body_region_attrs") {
       prop.body_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_region_attrs)>>(value);
       return;
    }

    if (name == "cond_attrs") {
       prop.cond_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond_attrs)>>(value);
       return;
    }

    if (name == "cond_region_attrs") {
       prop.cond_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond_region_attrs)>>(value);
       return;
    }

    if (name == "parallel_iterations") {
       prop.parallel_iterations = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.parallel_iterations)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void StatelessWhileRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.body_attrs) attrs.append("body_attrs", prop.body_attrs);

    if (prop.body_region_attrs) attrs.append("body_region_attrs", prop.body_region_attrs);

    if (prop.cond_attrs) attrs.append("cond_attrs", prop.cond_attrs);

    if (prop.cond_region_attrs) attrs.append("cond_region_attrs", prop.cond_region_attrs);

    if (prop.parallel_iterations) attrs.append("parallel_iterations", prop.parallel_iterations);
  attrs.append("operandSegmentSizes", ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes));
}

::mlir::LogicalResult StatelessWhileRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "body_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "body_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "cond_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "cond_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getParallelIterationsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops18(attr, "parallel_iterations", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult StatelessWhileRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.body_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.body_region_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cond_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cond_region_attrs)))
    return ::mlir::failure();

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  if (::mlir::failed(reader.readAttribute(prop.parallel_iterations)))
    return ::mlir::failure();

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void StatelessWhileRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.body_attrs);

  writer.writeOptionalAttribute(prop.body_region_attrs);

  writer.writeOptionalAttribute(prop.cond_attrs);

  writer.writeOptionalAttribute(prop.cond_region_attrs);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}
  writer.writeAttribute(prop.parallel_iterations);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

uint64_t StatelessWhileRegionOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > StatelessWhileRegionOp::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > StatelessWhileRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOp::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOp::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void StatelessWhileRegionOp::setParallelIterations(uint64_t attrValue) {
  getProperties().parallel_iterations = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<StatelessWhileRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult StatelessWhileRegionOp::verifyInvariantsImpl() {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_body_region_attrs = getProperties().body_region_attrs; (void)tblgen_body_region_attrs;
  auto tblgen_cond_attrs = getProperties().cond_attrs; (void)tblgen_cond_attrs;
  auto tblgen_cond_region_attrs = getProperties().cond_region_attrs; (void)tblgen_cond_region_attrs;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitOpError("requires attribute 'parallel_iterations'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_cond_attrs, "cond_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_cond_region_attrs, "cond_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_body_region_attrs, "body_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "cond_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessWhileRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatelessWhileRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> cond_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> initTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*cond_regionRegion))
    return ::mlir::failure();

  if (cond_regionRegion->empty()) cond_regionRegion->emplaceBlock();
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  ::mlir::FunctionType init__outs_functionType;
  if (parser.parseType(init__outs_functionType))
    return ::mlir::failure();
  initTypes = init__outs_functionType.getInputs();
  outsTypes = init__outs_functionType.getResults();
  }
  result.addRegion(std::move(cond_regionRegion));
  result.addRegion(std::move(body_regionRegion));
::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}), result.getOrAddProperties<StatelessWhileRegionOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessWhileRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getCondRegion());
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getInit().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printFunctionalType(getInit().getTypes(), getOuts().getTypes());
  }
}

StringRef StatelessWhileRegionOp::getDefaultDialect() { return "tfg"; }
void StatelessWhileRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

ConditionOp StatelessWhileRegionOp::getCondCondition() {
  return cast<ConditionOp>(getCondBlock().getTerminator());
}
YieldOp StatelessWhileRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}

OperandRange StatelessWhileRegionOp::getEntrySuccessorOperands(
     ::mlir::RegionBranchPoint point) {
  return getInit();
}
void StatelessWhileRegionOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return GetWhileLikeRegionOpSuccessorRegions(*this, point, regions);
}

LogicalResult StatelessWhileRegionOp::verify() { return VerifyWhileLikeRegionOp(*this); }

BlockArgument StatelessWhileRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatelessWhileRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatelessWhileRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatelessWhileRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessWhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::WhileOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WhileOpGenericAdaptorBase::WhileOpGenericAdaptorBase(WhileOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> WhileOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::tf_type::FuncAttr WhileOpGenericAdaptorBase::getCond() {
  auto attr = getCondAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr WhileOpGenericAdaptorBase::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

uint64_t WhileOpGenericAdaptorBase::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > WhileOpGenericAdaptorBase::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > WhileOpGenericAdaptorBase::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
WhileOpAdaptor::WhileOpAdaptor(WhileOp op) : WhileOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult WhileOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitError(loc, "'tfg.While' op ""requires attribute 'body'");
  auto tblgen_cond = getProperties().cond; (void)tblgen_cond;
  if (!tblgen_cond) return emitError(loc, "'tfg.While' op ""requires attribute 'cond'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitError(loc, "'tfg.While' op ""requires attribute 'parallel_iterations'");

  if (tblgen_cond && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_cond))))
    return emitError(loc, "'tfg.While' op ""attribute 'cond' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_body && !((::llvm::isa<::mlir::tf_type::FuncAttr>(tblgen_body))))
    return emitError(loc, "'tfg.While' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_parallel_iterations && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_parallel_iterations))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_parallel_iterations).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.While' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_T && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_T))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_T), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'tfg.While' op ""attribute 'T' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_output_shapes))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_output_shapes), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::tf_type::ShapeAttr>(attr))); }))))
    return emitError(loc, "'tfg.While' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void WhileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> WhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange WhileOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult WhileOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.T;
       auto attr = dict.get("T");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `T` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body;
       auto attr = dict.get("body");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond;
       auto attr = dict.get("cond");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_shapes;
       auto attr = dict.get("output_shapes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `output_shapes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.parallel_iterations;
       auto attr = dict.get("parallel_iterations");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `parallel_iterations` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute WhileOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.T;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("T",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_shapes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_shapes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.parallel_iterations;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("parallel_iterations",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WhileOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.T.getAsOpaquePointer()), 
    llvm::hash_value(prop.body.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_shapes.getAsOpaquePointer()), 
    llvm::hash_value(prop.parallel_iterations.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> WhileOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "T")
      return prop.T;

    if (name == "body")
      return prop.body;

    if (name == "cond")
      return prop.cond;

    if (name == "output_shapes")
      return prop.output_shapes;

    if (name == "parallel_iterations")
      return prop.parallel_iterations;
  return std::nullopt;
}

void WhileOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "T") {
       prop.T = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.T)>>(value);
       return;
    }

    if (name == "body") {
       prop.body = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body)>>(value);
       return;
    }

    if (name == "cond") {
       prop.cond = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond)>>(value);
       return;
    }

    if (name == "output_shapes") {
       prop.output_shapes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_shapes)>>(value);
       return;
    }

    if (name == "parallel_iterations") {
       prop.parallel_iterations = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.parallel_iterations)>>(value);
       return;
    }
}

void WhileOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.T) attrs.append("T", prop.T);

    if (prop.body) attrs.append("body", prop.body);

    if (prop.cond) attrs.append("cond", prop.cond);

    if (prop.output_shapes) attrs.append("output_shapes", prop.output_shapes);

    if (prop.parallel_iterations) attrs.append("parallel_iterations", prop.parallel_iterations);
}

::mlir::LogicalResult WhileOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops1(attr, "T", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "body", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops5(attr, "cond", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputShapesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops2(attr, "output_shapes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getParallelIterationsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops18(attr, "parallel_iterations", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult WhileOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.T)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.body)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.cond)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.output_shapes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.parallel_iterations)))
    return ::mlir::failure();
  return ::mlir::success();
}

void WhileOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.T);
  writer.writeAttribute(prop.body);
  writer.writeAttribute(prop.cond);

  writer.writeOptionalAttribute(prop.output_shapes);
  writer.writeAttribute(prop.parallel_iterations);
}

::mlir::tf_type::FuncAttr WhileOp::getCond() {
  auto attr = getCondAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

::mlir::tf_type::FuncAttr WhileOp::getBody() {
  auto attr = getBodyAttr();
  return ::llvm::cast<::mlir::tf_type::FuncAttr>(attr);
}

uint64_t WhileOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > WhileOp::getT() {
  auto attr = getTAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > WhileOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void WhileOp::setParallelIterations(uint64_t attrValue) {
  getProperties().parallel_iterations = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations) {
      build(odsBuilder, odsState, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().cond = cond;
  odsState.getOrAddProperties<Properties>().body = body;
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (T) {
    odsState.getOrAddProperties<Properties>().T = T;
  }
  if (output_shapes) {
    odsState.getOrAddProperties<Properties>().output_shapes = output_shapes;
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WhileOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult WhileOp::verifyInvariantsImpl() {
  auto tblgen_T = getProperties().T; (void)tblgen_T;
  auto tblgen_body = getProperties().body; (void)tblgen_body;
  if (!tblgen_body) return emitOpError("requires attribute 'body'");
  auto tblgen_cond = getProperties().cond; (void)tblgen_cond;
  if (!tblgen_cond) return emitOpError("requires attribute 'cond'");
  auto tblgen_output_shapes = getProperties().output_shapes; (void)tblgen_output_shapes;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitOpError("requires attribute 'parallel_iterations'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_cond, "cond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult WhileOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyWhileLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::WhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::WhileRegionOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
WhileRegionOpGenericAdaptorBase::WhileRegionOpGenericAdaptorBase(WhileRegionOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> WhileRegionOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

uint64_t WhileRegionOpGenericAdaptorBase::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > WhileRegionOpGenericAdaptorBase::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > WhileRegionOpGenericAdaptorBase::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> WhileRegionOpGenericAdaptorBase::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> WhileRegionOpGenericAdaptorBase::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
WhileRegionOpAdaptor::WhileRegionOpAdaptor(WhileRegionOp op) : WhileRegionOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult WhileRegionOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_body_region_attrs = getProperties().body_region_attrs; (void)tblgen_body_region_attrs;
  auto tblgen_cond_attrs = getProperties().cond_attrs; (void)tblgen_cond_attrs;
  auto tblgen_cond_region_attrs = getProperties().cond_region_attrs; (void)tblgen_cond_region_attrs;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitError(loc, "'tfg.WhileRegion' op ""requires attribute 'parallel_iterations'");

  if (tblgen_parallel_iterations && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_parallel_iterations))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_parallel_iterations).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_cond_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_cond_attrs))))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'cond_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_body_attrs && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_body_attrs))))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_cond_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_cond_region_attrs))))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'cond_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_body_region_attrs && !((::llvm::isa<::mlir::tfg::RegionAttr>(tblgen_body_region_attrs))))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'body_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WhileRegionOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange WhileRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange WhileRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> WhileRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::LogicalResult WhileRegionOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.body_attrs;
       auto attr = dict.get("body_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.body_region_attrs;
       auto attr = dict.get("body_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `body_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond_attrs;
       auto attr = dict.get("cond_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.cond_region_attrs;
       auto attr = dict.get("cond_region_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `cond_region_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.parallel_iterations;
       auto attr = dict.get("parallel_iterations");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `parallel_iterations` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute WhileRegionOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.body_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.body_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("body_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.cond_region_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("cond_region_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.parallel_iterations;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("parallel_iterations",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes",
                                              ::mlir::DenseI32ArrayAttr::get(ctx, propStorage)));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WhileRegionOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.body_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.body_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.cond_region_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.parallel_iterations.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> WhileRegionOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "body_attrs")
      return prop.body_attrs;

    if (name == "body_region_attrs")
      return prop.body_region_attrs;

    if (name == "cond_attrs")
      return prop.cond_attrs;

    if (name == "cond_region_attrs")
      return prop.cond_region_attrs;

    if (name == "parallel_iterations")
      return prop.parallel_iterations;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes);
  return std::nullopt;
}

void WhileRegionOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "body_attrs") {
       prop.body_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_attrs)>>(value);
       return;
    }

    if (name == "body_region_attrs") {
       prop.body_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.body_region_attrs)>>(value);
       return;
    }

    if (name == "cond_attrs") {
       prop.cond_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond_attrs)>>(value);
       return;
    }

    if (name == "cond_region_attrs") {
       prop.cond_region_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.cond_region_attrs)>>(value);
       return;
    }

    if (name == "parallel_iterations") {
       prop.parallel_iterations = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.parallel_iterations)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void WhileRegionOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.body_attrs) attrs.append("body_attrs", prop.body_attrs);

    if (prop.body_region_attrs) attrs.append("body_region_attrs", prop.body_region_attrs);

    if (prop.cond_attrs) attrs.append("cond_attrs", prop.cond_attrs);

    if (prop.cond_region_attrs) attrs.append("cond_region_attrs", prop.cond_region_attrs);

    if (prop.parallel_iterations) attrs.append("parallel_iterations", prop.parallel_iterations);
  attrs.append("operandSegmentSizes", ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes));
}

::mlir::LogicalResult WhileRegionOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBodyAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "body_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getBodyRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "body_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops6(attr, "cond_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCondRegionAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops7(attr, "cond_region_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getParallelIterationsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ops18(attr, "parallel_iterations", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult WhileRegionOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.body_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.body_region_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cond_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.cond_region_attrs)))
    return ::mlir::failure();

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  if (::mlir::failed(reader.readAttribute(prop.parallel_iterations)))
    return ::mlir::failure();

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void WhileRegionOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.body_attrs);

  writer.writeOptionalAttribute(prop.body_region_attrs);

  writer.writeOptionalAttribute(prop.cond_attrs);

  writer.writeOptionalAttribute(prop.cond_region_attrs);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}
  writer.writeAttribute(prop.parallel_iterations);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

uint64_t WhileRegionOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > WhileRegionOp::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::DictionaryAttr > WhileRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> WhileRegionOp::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

::std::optional<::mlir::tfg::RegionAttr> WhileRegionOp::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::std::optional<::mlir::tfg::RegionAttr>(::llvm::cast<::mlir::tfg::RegionAttr>(attr)) : (::std::nullopt);
}

void WhileRegionOp::setParallelIterations(uint64_t attrValue) {
  getProperties().parallel_iterations = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = parallel_iterations;
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().parallel_iterations = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations);
  if (cond_attrs) {
    odsState.getOrAddProperties<Properties>().cond_attrs = cond_attrs;
  }
  if (body_attrs) {
    odsState.getOrAddProperties<Properties>().body_attrs = body_attrs;
  }
  if (cond_region_attrs) {
    odsState.getOrAddProperties<Properties>().cond_region_attrs = cond_region_attrs;
  }
  if (body_region_attrs) {
    odsState.getOrAddProperties<Properties>().body_region_attrs = body_region_attrs;
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WhileRegionOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult WhileRegionOp::verifyInvariantsImpl() {
  auto tblgen_body_attrs = getProperties().body_attrs; (void)tblgen_body_attrs;
  auto tblgen_body_region_attrs = getProperties().body_region_attrs; (void)tblgen_body_region_attrs;
  auto tblgen_cond_attrs = getProperties().cond_attrs; (void)tblgen_cond_attrs;
  auto tblgen_cond_region_attrs = getProperties().cond_region_attrs; (void)tblgen_cond_region_attrs;
  auto tblgen_parallel_iterations = getProperties().parallel_iterations; (void)tblgen_parallel_iterations;
  if (!tblgen_parallel_iterations) return emitOpError("requires attribute 'parallel_iterations'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_cond_attrs, "cond_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_cond_region_attrs, "cond_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_body_region_attrs, "body_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "cond_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult WhileRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WhileRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> cond_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> initTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*cond_regionRegion))
    return ::mlir::failure();

  if (cond_regionRegion->empty()) cond_regionRegion->emplaceBlock();
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  ::mlir::FunctionType init__outs_functionType;
  if (parser.parseType(init__outs_functionType))
    return ::mlir::failure();
  initTypes = init__outs_functionType.getInputs();
  outsTypes = init__outs_functionType.getResults();
  }
  result.addRegion(std::move(cond_regionRegion));
  result.addRegion(std::move(body_regionRegion));
::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}), result.getOrAddProperties<WhileRegionOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WhileRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getCondRegion());
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getInit().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printFunctionalType(getInit().getTypes(), getOuts().getTypes());
  }
}

StringRef WhileRegionOp::getDefaultDialect() { return "tfg"; }
void WhileRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

ConditionOp WhileRegionOp::getCondCondition() {
  return cast<ConditionOp>(getCondBlock().getTerminator());
}
YieldOp WhileRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}

OperandRange WhileRegionOp::getEntrySuccessorOperands(
     ::mlir::RegionBranchPoint point) {
  return getInit();
}
void WhileRegionOp::getSuccessorRegions(
    ::mlir::RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return GetWhileLikeRegionOpSuccessorRegions(*this, point, regions);
}

LogicalResult WhileRegionOp::verify() { return VerifyWhileLikeRegionOp(*this); }

BlockArgument WhileRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument WhileRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument WhileRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument WhileRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::WhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::YieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
YieldOpGenericAdaptorBase::YieldOpGenericAdaptorBase(YieldOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> YieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

} // namespace detail
YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : YieldOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange YieldOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange YieldOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::LogicalResult YieldOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
    {

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute YieldOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.operandSegmentSizes;
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes",
                                              ::mlir::DenseI32ArrayAttr::get(ctx, propStorage)));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code YieldOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> YieldOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes);
  return std::nullopt;
}

void YieldOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void YieldOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
  attrs.append("operandSegmentSizes", ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes));
}

::mlir::LogicalResult YieldOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

::mlir::LogicalResult YieldOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void YieldOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<YieldOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  }
::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(argsOperands.size()), static_cast<int32_t>(ctlsOperands.size())}), result.getOrAddProperties<YieldOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getArgs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getArgs().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
  }
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::YieldOp)


#endif  // GET_OP_CLASSES

