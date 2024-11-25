//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include <vtkm/cont/UncertainArrayHandle.h>
#include <vtkm/cont/UnknownArrayHandle.h>

#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/ArrayHandleConstant.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/ArrayHandleGroupVecVariable.h>
#include <vtkm/cont/ArrayHandleMultiplexer.h>
#include <vtkm/cont/ArrayHandleRuntimeVec.h>

#include <vtkm/TypeTraits.h>

#include <vtkm/cont/testing/Testing.h>

namespace
{

// Make an "unusual" type to use in the test. This is simply a type that
// is sure not to be declared elsewhere.
struct UnusualType
{
  using T = vtkm::Id;
  T X;
  UnusualType() = default;
  UnusualType(T x)
    : X(x)
  {
  }
  UnusualType& operator=(T x)
  {
    this->X = x;
    return *this;
  }
  operator T() const { return this->X; }
};

const vtkm::Id ARRAY_SIZE = 10;

struct CheckFunctor
{
  template <typename T, typename S>
  static void CheckArray(const vtkm::cont::ArrayHandle<T, S>& array)
  {
    VTKM_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE, "Unexpected array size.");
    CheckPortal(array.ReadPortal());
  }

  template <typename S>
  static void CheckArray(const vtkm::cont::ArrayHandle<UnusualType, S>& array)
  {
    VTKM_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE, "Unexpected array size.");
    auto portal = array.ReadPortal();
    for (vtkm::Id index = 0; index < array.GetNumberOfValues(); ++index)
    {
      VTKM_TEST_ASSERT(portal.Get(index) == TestValue(index, UnusualType::T{}));
    }
  }

  template <typename T, typename S>
  void operator()(const vtkm::cont::ArrayHandle<T, S>& array, bool& called) const
  {
    called = true;
    std::cout << "  Checking for array type " << vtkm::cont::TypeToString<T>() << " with storage "
              << vtkm::cont::TypeToString<S>() << std::endl;

    CheckArray(array);
  }
};

void BasicUnknownArrayChecks(const vtkm::cont::UnknownArrayHandle& array,
                             vtkm::IdComponent numComponents)
{
  std::cout << "  Checking an UnknownArrayHandle containing " << array.GetArrayTypeName()
            << std::endl;
  VTKM_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE,
                   "Dynamic array reports unexpected size.");
  VTKM_TEST_ASSERT(array.GetNumberOfComponentsFlat() == numComponents,
                   "Dynamic array reports unexpected number of components.");
}

void CheckUnknownArrayDefaults(const vtkm::cont::UnknownArrayHandle& array,
                               vtkm::IdComponent numComponents)
{
  BasicUnknownArrayChecks(array, numComponents);

  std::cout << "  CastAndCall with default types" << std::endl;
  bool called = false;
  vtkm::cont::CastAndCall(array, CheckFunctor(), called);
}

template <typename TypeList, typename StorageList>
void CheckUnknownArray(const vtkm::cont::UnknownArrayHandle& array, vtkm::IdComponent numComponents)
{
  VTKM_IS_LIST(TypeList);
  VTKM_IS_LIST(StorageList);

  BasicUnknownArrayChecks(array, numComponents);

  std::cout << "  CastAndCall with given types" << std::endl;
  bool called = false;
  array.CastAndCallForTypes<TypeList, StorageList>(CheckFunctor{}, called);
  VTKM_TEST_ASSERT(
    called, "The functor was never called (and apparently a bad value exception not thrown).");

  std::cout << "  Check CastAndCall again with UncertainArrayHandle" << std::endl;
  called = false;
  vtkm::cont::CastAndCall(array.ResetTypes<TypeList, StorageList>(), CheckFunctor{}, called);
  VTKM_TEST_ASSERT(
    called, "The functor was never called (and apparently a bad value exception not thrown).");
}

template <typename T>
vtkm::cont::ArrayHandle<T> CreateArray(T)
{
  vtkm::cont::ArrayHandle<T> array;
  array.Allocate(ARRAY_SIZE);
  SetPortal(array.WritePortal());
  return array;
}

vtkm::cont::ArrayHandle<UnusualType> CreateArray(UnusualType)
{
  vtkm::cont::ArrayHandle<UnusualType> array;
  array.Allocate(ARRAY_SIZE);
  auto portal = array.WritePortal();
  for (vtkm::Id index = 0; index < ARRAY_SIZE; ++index)
  {
    portal.Set(index, TestValue(index, UnusualType::T{}));
  }
  return array;
}

template <typename T>
vtkm::cont::UnknownArrayHandle CreateArrayUnknown(T t)
{
  return vtkm::cont::UnknownArrayHandle(CreateArray(t));
}

template <typename ArrayHandleType>
void CheckAsArrayHandle(const ArrayHandleType& array)
{
  VTKM_IS_ARRAY_HANDLE(ArrayHandleType);
  using T = typename ArrayHandleType::ValueType;

  vtkm::cont::UnknownArrayHandle arrayUnknown = array;
  VTKM_TEST_ASSERT(!arrayUnknown.IsType<vtkm::cont::ArrayHandle<UnusualType>>(),
                   "Dynamic array reporting is wrong type.");

  {
    std::cout << "    Normal get ArrayHandle" << std::endl;
    ArrayHandleType retreivedArray1;
    arrayUnknown.AsArrayHandle(retreivedArray1);
    VTKM_TEST_ASSERT(arrayUnknown.CanConvert<ArrayHandleType>(), "Did not query handle correctly.");
    VTKM_TEST_ASSERT(array == retreivedArray1, "Did not get back same array.");

    ArrayHandleType retreivedArray2 = arrayUnknown.AsArrayHandle<ArrayHandleType>();
    VTKM_TEST_ASSERT(array == retreivedArray2, "Did not get back same array.");
  }

  {
    std::cout << "    Put in cast array, get actual array" << std::endl;
    auto castArray = vtkm::cont::make_ArrayHandleCast<vtkm::Float64>(array);
    vtkm::cont::UnknownArrayHandle arrayUnknown2(castArray);
    VTKM_TEST_ASSERT(arrayUnknown2.IsType<ArrayHandleType>());
    ArrayHandleType retrievedArray = arrayUnknown2.AsArrayHandle<ArrayHandleType>();
    VTKM_TEST_ASSERT(array == retrievedArray);
  }

  {
    std::cout << "    Get array as cast" << std::endl;
    vtkm::cont::ArrayHandleCast<vtkm::Float64, ArrayHandleType> castArray;
    arrayUnknown.AsArrayHandle(castArray);
    VTKM_TEST_ASSERT(test_equal_portals(array.ReadPortal(), castArray.ReadPortal()));
  }

  {
    std::cout << "    Put in multiplexer, get actual array" << std::endl;
    vtkm::cont::UnknownArrayHandle arrayUnknown2 = vtkm::cont::ArrayHandleMultiplexer<
      ArrayHandleType,
      vtkm::cont::ArrayHandleConstant<typename ArrayHandleType::ValueType>>(array);
    VTKM_TEST_ASSERT(arrayUnknown2.IsType<ArrayHandleType>(),
                     "Putting in multiplexer did not pull out array.");
  }

  {
    std::cout << "    Make sure multiplex array prefers direct array (1st arg)" << std::endl;
    using MultiplexerType =
      vtkm::cont::ArrayHandleMultiplexer<ArrayHandleType,
                                         vtkm::cont::ArrayHandleCast<T, ArrayHandleType>>;
    MultiplexerType multiplexArray = arrayUnknown.AsArrayHandle<MultiplexerType>();

    VTKM_TEST_ASSERT(multiplexArray.IsValid());
    VTKM_TEST_ASSERT(multiplexArray.GetArrayHandleVariant().GetIndex() == 0);
    VTKM_TEST_ASSERT(test_equal_portals(multiplexArray.ReadPortal(), array.ReadPortal()));
  }

  {
    std::cout << "    Make sure multiplex array prefers direct array (2nd arg)" << std::endl;
    using MultiplexerType =
      vtkm::cont::ArrayHandleMultiplexer<vtkm::cont::ArrayHandleCast<T, vtkm::cont::ArrayHandle<T>>,
                                         ArrayHandleType>;
    MultiplexerType multiplexArray = arrayUnknown.AsArrayHandle<MultiplexerType>();

    VTKM_TEST_ASSERT(multiplexArray.IsValid());
    VTKM_TEST_ASSERT(multiplexArray.GetArrayHandleVariant().GetIndex() == 1);
    VTKM_TEST_ASSERT(test_equal_portals(multiplexArray.ReadPortal(), array.ReadPortal()));
  }

  {
    std::cout << "    Make sure adding arrays follows nesting of special arrays" << std::endl;
    vtkm::cont::ArrayHandleMultiplexer<vtkm::cont::ArrayHandle<vtkm::Int64>,
                                       vtkm::cont::ArrayHandleCast<vtkm::Int64, ArrayHandleType>>
      multiplexer(vtkm::cont::make_ArrayHandleCast<vtkm::Int64>(array));
    auto crazyArray = vtkm::cont::make_ArrayHandleCast<vtkm::Float64>(multiplexer);
    vtkm::cont::UnknownArrayHandle arrayUnknown2(crazyArray);
    VTKM_TEST_ASSERT(arrayUnknown2.IsType<ArrayHandleType>());
    ArrayHandleType retrievedArray = arrayUnknown2.AsArrayHandle<ArrayHandleType>();
    VTKM_TEST_ASSERT(array == retrievedArray);
  }

  {
    std::cout << "    Try adding arrays with variable amounts of components" << std::endl;
    // There might be some limited functionality, but you should still be able
    // to get arrays in and out.

    // Note, this is a bad way to implement this array. You should something like
    // ArrayHandleGroupVec instead.
    using VariableVecArrayType =
      vtkm::cont::ArrayHandleGroupVecVariable<ArrayHandleType,
                                              vtkm::cont::ArrayHandleCounting<vtkm::Id>>;
    VariableVecArrayType inArray = vtkm::cont::make_ArrayHandleGroupVecVariable(
      array, vtkm::cont::make_ArrayHandleCounting<vtkm::Id>(0, 2, ARRAY_SIZE / 2 + 1));
    VTKM_TEST_ASSERT(inArray.GetNumberOfValues() == ARRAY_SIZE / 2);
    vtkm::cont::UnknownArrayHandle arrayUnknown2 = inArray;
    VTKM_TEST_ASSERT(arrayUnknown2.IsType<VariableVecArrayType>());
    VariableVecArrayType retrievedArray = arrayUnknown2.AsArrayHandle<VariableVecArrayType>();
    VTKM_TEST_ASSERT(retrievedArray == inArray);
  }
}

// A vtkm::Vec if NumComps > 1, otherwise a scalar
template <typename T, vtkm::IdComponent NumComps>
using VecOrScalar = typename std::conditional<(NumComps > 1), vtkm::Vec<T, NumComps>, T>::type;

template <typename T>
void TryNewInstance(vtkm::cont::UnknownArrayHandle originalArray)
{
  // This check should already have been performed by caller, but just in case.
  CheckUnknownArray<vtkm::List<T>, VTKM_DEFAULT_STORAGE_LIST>(originalArray,
                                                              vtkm::VecTraits<T>::NUM_COMPONENTS);

  std::cout << "Create new instance of array." << std::endl;
  vtkm::cont::UnknownArrayHandle newArray = originalArray.NewInstance();

  std::cout << "Get a static instance of the new array (which checks the type)." << std::endl;
  vtkm::cont::ArrayHandle<T> staticArray;
  newArray.AsArrayHandle(staticArray);

  std::cout << "Fill the new array with invalid values and make sure the original" << std::endl
            << "is uneffected." << std::endl;
  staticArray.Allocate(ARRAY_SIZE);
  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    staticArray.WritePortal().Set(index, TestValue(index + 100, T()));
  }
  CheckUnknownArray<vtkm::List<T>, VTKM_DEFAULT_STORAGE_LIST>(originalArray,
                                                              vtkm::VecTraits<T>::NUM_COMPONENTS);

  std::cout << "Set the new static array to expected values and make sure the new" << std::endl
            << "dynamic array points to the same new values." << std::endl;
  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    staticArray.WritePortal().Set(index, TestValue(index, T()));
  }
  CheckUnknownArray<vtkm::List<T>, VTKM_DEFAULT_STORAGE_LIST>(newArray,
                                                              vtkm::VecTraits<T>::NUM_COMPONENTS);

  std::cout << "Get a new instance as a float array and make sure the type is as expected."
            << std::endl;
  vtkm::cont::UnknownArrayHandle floatArray = originalArray.NewInstanceFloatBasic();
  vtkm::cont::ArrayHandle<
    typename vtkm::VecTraits<T>::template ReplaceBaseComponentType<vtkm::FloatDefault>>
    staticFloatArray;
  floatArray.AsArrayHandle(staticFloatArray);
}

template <typename ActualT>
struct CheckActualTypeFunctor
{
  template <typename T, typename S>
  void operator()(const vtkm::cont::ArrayHandle<T, S>& array, bool& called) const
  {
    called = true;
    VTKM_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE, "Unexpected array size.");
    auto portal = array.ReadPortal();
    for (vtkm::Id i = 0; i < ARRAY_SIZE; ++i)
    {
      T retrieved = portal.Get(i);
      ActualT expected = TestValue(i, ActualT{});
      VTKM_TEST_ASSERT(test_equal(retrieved, expected));
    }
  }
};

template <typename T>
void TryCastAndCallFallback()
{
  vtkm::cont::UnknownArrayHandle array = CreateArrayUnknown(T{});

  using FallbackTypes = vtkm::List<vtkm::FloatDefault,
                                   vtkm::Vec2f,
                                   vtkm::Vec3f,
                                   vtkm::Vec4f,
                                   vtkm::Vec<vtkm::Vec2f, 3>,
                                   vtkm::Vec<vtkm::Vec<vtkm::Vec4f, 3>, 2>>;
  bool called = false;
  array.CastAndCallForTypesWithFloatFallback<FallbackTypes, vtkm::cont::StorageListBasic>(
    CheckActualTypeFunctor<T>{}, called);
  VTKM_TEST_ASSERT(
    called, "The functor was never called (and apparently a bad value exception not thrown).");
}

void TryCastAndCallFallback()
{
  std::cout << "  Scalar array." << std::endl;
  TryCastAndCallFallback<vtkm::Float64>();

  std::cout << "  Equivalent scalar." << std::endl;
  TryCastAndCallFallback<VTKM_UNUSED_INT_TYPE>();

  std::cout << "  Basic Vec." << std::endl;
  TryCastAndCallFallback<vtkm::Id3>();

  std::cout << "  Vec of Vecs." << std::endl;
  TryCastAndCallFallback<vtkm::Vec<vtkm::Vec2f_32, 3>>();

  std::cout << "  Vec of Vecs of Vecs." << std::endl;
  TryCastAndCallFallback<vtkm::Vec<vtkm::Vec<vtkm::Id4, 3>, 2>>();
}

template <typename T>
void TryAsMultiplexer(vtkm::cont::UnknownArrayHandle sourceArray)
{
  auto originalArray = sourceArray.AsArrayHandle<vtkm::cont::ArrayHandle<T>>();

  {
    std::cout << "Get multiplex array through direct type." << std::endl;
    using MultiplexerType = vtkm::cont::ArrayHandleMultiplexer<vtkm::cont::ArrayHandle<T>,
                                                               vtkm::cont::ArrayHandleConstant<T>>;
    VTKM_TEST_ASSERT(sourceArray.CanConvert<MultiplexerType>());
    MultiplexerType multiplexArray = sourceArray.AsArrayHandle<MultiplexerType>();

    VTKM_TEST_ASSERT(multiplexArray.IsValid());
    VTKM_TEST_ASSERT(test_equal_portals(multiplexArray.ReadPortal(), originalArray.ReadPortal()));
  }

  {
    std::cout << "Get multiplex array through cast type." << std::endl;
    using CastT = typename vtkm::VecTraits<T>::template ReplaceBaseComponentType<vtkm::Float64>;
    using MultiplexerType = vtkm::cont::ArrayHandleMultiplexer<
      vtkm::cont::ArrayHandle<CastT>,
      vtkm::cont::ArrayHandleCast<CastT, vtkm::cont::ArrayHandle<T>>>;
    VTKM_TEST_ASSERT(sourceArray.CanConvert<MultiplexerType>());
    MultiplexerType multiplexArray = sourceArray.AsArrayHandle<MultiplexerType>();

    VTKM_TEST_ASSERT(multiplexArray.IsValid());
    VTKM_TEST_ASSERT(test_equal_portals(multiplexArray.ReadPortal(), originalArray.ReadPortal()));
  }

#if 0
  // Maybe we should support this, but right now we don't
  {
    std::cout << "Make sure multiplex array prefers direct array (1st arg)" << std::endl;
    using MultiplexerType = vtkm::cont::ArrayHandleMultiplexer<
      vtkm::cont::ArrayHandle<T>,
      vtkm::cont::ArrayHandleCast<T, vtkm::cont::ArrayHandle<T>>>;
    MultiplexerType multiplexArray = sourceArray.AsArrayHandle<MultiplexerType>();

    VTKM_TEST_ASSERT(multiplexArray.IsValid());
    VTKM_TEST_ASSERT(multiplexArray.GetStorage().GetArrayHandleVariant().GetIndex() == 0);
    VTKM_TEST_ASSERT(test_equal_portals(multiplexArray.ReadPortal(), originalArray.ReadPortal()));
  }

  {
    std::cout << "Make sure multiplex array prefers direct array (2nd arg)" << std::endl;
    using MultiplexerType =
      vtkm::cont::ArrayHandleMultiplexer<vtkm::cont::ArrayHandleCast<T, vtkm::cont::ArrayHandle<T>>,
                                         vtkm::cont::ArrayHandle<T>>;
    MultiplexerType multiplexArray = sourceArray.AsArrayHandle<MultiplexerType>();

    VTKM_TEST_ASSERT(multiplexArray.IsValid());
    VTKM_TEST_ASSERT(multiplexArray.GetStorage().GetArrayHandleVariant().GetIndex() == 1);
    VTKM_TEST_ASSERT(test_equal_portals(multiplexArray.ReadPortal(), originalArray.ReadPortal()));
  }
#endif
}

struct SimpleRecombineCopy
{
  template <typename T>
  void operator()(const vtkm::cont::ArrayHandleRecombineVec<T>& inputArray,
                  const vtkm::cont::UnknownArrayHandle& output) const
  {
    vtkm::cont::ArrayHandleRecombineVec<T> outputArray =
      output.ExtractArrayFromComponents<T>(vtkm::CopyFlag::Off);
    vtkm::Id size = inputArray.GetNumberOfValues();
    outputArray.Allocate(size);
    auto inputPortal = inputArray.ReadPortal();
    auto outputPortal = outputArray.WritePortal();

    for (vtkm::Id index = 0; index < size; ++index)
    {
      outputPortal.Set(index, inputPortal.Get(index));
    }
  }
};

template <typename T>
void TryExtractArray(const vtkm::cont::UnknownArrayHandle& originalArray)
{
  // This check should already have been performed by caller, but just in case.
  CheckUnknownArray<vtkm::List<T>, VTKM_DEFAULT_STORAGE_LIST>(originalArray,
                                                              vtkm::VecTraits<T>::NUM_COMPONENTS);

  std::cout << "Create new instance of array." << std::endl;
  vtkm::cont::UnknownArrayHandle newArray = originalArray.NewInstanceBasic();

  std::cout << "Do CastAndCallWithExtractedArray." << std::endl;
  originalArray.CastAndCallWithExtractedArray(SimpleRecombineCopy{}, newArray);

  CheckUnknownArray<vtkm::List<T>, VTKM_DEFAULT_STORAGE_LIST>(newArray,
                                                              vtkm::VecTraits<T>::NUM_COMPONENTS);
}

template <typename T>
void TryDefaultType()
{
  vtkm::cont::UnknownArrayHandle array = CreateArrayUnknown(T{});

  CheckUnknownArrayDefaults(array, vtkm::VecTraits<T>::NUM_COMPONENTS);

  TryNewInstance<T>(array);

  TryAsMultiplexer<T>(array);

  TryExtractArray<T>(array);
}

struct TryBasicVTKmType
{
  template <typename T>
  void operator()(T) const
  {
    vtkm::cont::UnknownArrayHandle array = CreateArrayUnknown(T());

    VTKM_TEST_ASSERT(array.GetValueTypeName() == vtkm::cont::TypeToString<T>());
    VTKM_TEST_ASSERT(array.GetStorageTypeName() ==
                     vtkm::cont::TypeToString<vtkm::cont::StorageTagBasic>());

    CheckUnknownArray<vtkm::TypeListAll, VTKM_DEFAULT_STORAGE_LIST>(
      array, vtkm::VecTraits<T>::NUM_COMPONENTS);

    TryNewInstance<T>(array);
  }
};

void TryUnusualType()
{
  // A string is an unlikely type to be declared elsewhere in VTK-m.
  vtkm::cont::UnknownArrayHandle array = CreateArrayUnknown(UnusualType{});

  try
  {
    CheckUnknownArray<VTKM_DEFAULT_TYPE_LIST, VTKM_DEFAULT_STORAGE_LIST>(array, 1);
    VTKM_TEST_FAIL("CastAndCall failed to error for unrecognized type.");
  }
  catch (vtkm::cont::ErrorBadType&)
  {
    std::cout << "  Caught exception for unrecognized type." << std::endl;
  }
  CheckUnknownArray<vtkm::List<UnusualType>, VTKM_DEFAULT_STORAGE_LIST>(array, 1);
  std::cout << "  Found type when type list was reset." << std::endl;
}

template <typename ArrayHandleType>
void TryAsArrayHandle(const ArrayHandleType& array)
{
  CheckAsArrayHandle(array);
}

void TryAsArrayHandle()
{
  std::cout << "  Normal array handle." << std::endl;
  vtkm::Id buffer[ARRAY_SIZE];
  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    buffer[index] = TestValue(index, vtkm::Id());
  }

  vtkm::cont::ArrayHandle<vtkm::Id> array =
    vtkm::cont::make_ArrayHandle(buffer, ARRAY_SIZE, vtkm::CopyFlag::On);
  TryAsArrayHandle(array);

  std::cout << "  Constant array handle." << std::endl;
  TryAsArrayHandle(vtkm::cont::make_ArrayHandleConstant(5, ARRAY_SIZE));
}

struct CheckExtractedArray
{
  template <typename ExtractedArray, typename OriginalArray>
  void operator()(const ExtractedArray& extractedArray, const OriginalArray& originalArray) const
  {
    using ValueType = typename OriginalArray::ValueType;
    using FlatVec = vtkm::VecFlat<ValueType>;

    VTKM_TEST_ASSERT(extractedArray.GetNumberOfComponents() == FlatVec::NUM_COMPONENTS);
    auto originalPortal = originalArray.ReadPortal();
    auto extractedPortal = extractedArray.ReadPortal();
    for (vtkm::Id valueIndex = 0; valueIndex < ARRAY_SIZE; ++valueIndex)
    {
      FlatVec originalData = originalPortal.Get(valueIndex);
      auto extractedData = extractedPortal.Get(valueIndex);
      VTKM_TEST_ASSERT(test_equal(originalData, extractedData));
    }

    // Make sure an extracted array stuffed back into an UnknownArrayHandle works.
    // This can happen when working with an extracted array that is passed to functions
    // that are implemented with UnknownArrayHandle.
    vtkm::cont::UnknownArrayHandle unknownArray{ extractedArray };

    using ComponentType =
      typename vtkm::VecTraits<typename ExtractedArray::ValueType>::BaseComponentType;
    vtkm::cont::UnknownArrayHandle newBasic = unknownArray.NewInstanceBasic();
    newBasic.AsArrayHandle<vtkm::cont::ArrayHandleRuntimeVec<ComponentType>>();
    vtkm::cont::UnknownArrayHandle newFloat = unknownArray.NewInstanceFloatBasic();
    newFloat.AsArrayHandle<vtkm::cont::ArrayHandleRuntimeVec<vtkm::FloatDefault>>();
  }
};

template <typename ArrayHandleType>
void TryExtractComponent()
{
  using ValueType = typename ArrayHandleType::ValueType;
  using FlatVec = vtkm::VecFlat<ValueType>;
  using ComponentType = typename FlatVec::ComponentType;

  ArrayHandleType originalArray;
  originalArray.Allocate(ARRAY_SIZE);
  SetPortal(originalArray.WritePortal());

  vtkm::cont::UnknownArrayHandle unknownArray(originalArray);

  VTKM_TEST_ASSERT(unknownArray.GetNumberOfComponentsFlat() == FlatVec::NUM_COMPONENTS);

  CheckExtractedArray{}(unknownArray.ExtractArrayFromComponents<ComponentType>(), originalArray);

  unknownArray.CastAndCallWithExtractedArray(CheckExtractedArray{}, originalArray);
}

void TryExtractComponent()
{
  std::cout << "  Scalar array." << std::endl;
  TryExtractComponent<vtkm::cont::ArrayHandle<vtkm::FloatDefault>>();

  std::cout << "  Equivalent scalar." << std::endl;
  TryExtractComponent<vtkm::cont::ArrayHandle<VTKM_UNUSED_INT_TYPE>>();

  std::cout << "  Basic Vec." << std::endl;
  TryExtractComponent<vtkm::cont::ArrayHandle<vtkm::Id3>>();

  std::cout << "  Vec of Vecs." << std::endl;
  TryExtractComponent<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Vec2f, 3>>>();

  std::cout << "  Vec of Vecs of Vecs." << std::endl;
  TryExtractComponent<vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Vec<vtkm::Id4, 3>, 2>>>();
}

void TrySetCastArray()
{
  vtkm::cont::ArrayHandle<vtkm::Id> knownArray = CreateArray(vtkm::Id{});
  vtkm::cont::UnknownArrayHandle unknownArray(
    vtkm::cont::make_ArrayHandleCast<vtkm::Float32>(knownArray));

  // The unknownArray should actually hold the original knownArray type even though we gave it
  // a cast array.
  CheckUnknownArray<vtkm::List<vtkm::Id>, vtkm::List<VTKM_DEFAULT_STORAGE_TAG>>(unknownArray, 1);
}

void TrySetMultiplexerArray()
{
  vtkm::cont::ArrayHandle<vtkm::Id> knownArray = CreateArray(vtkm::Id{});
  vtkm::cont::ArrayHandleMultiplexer<vtkm::cont::ArrayHandle<vtkm::Id>,
                                     vtkm::cont::ArrayHandleConstant<vtkm::Id>>
    multiplexerArray(knownArray);
  vtkm::cont::UnknownArrayHandle unknownArray(multiplexerArray);

  // The unknownArray should actually hold the original knownArray type even though we gave it
  // a multiplexer array.
  CheckUnknownArray<vtkm::List<vtkm::Id>, vtkm::List<VTKM_DEFAULT_STORAGE_TAG>>(unknownArray, 1);
}

template <typename T, typename BasicComponentType = typename vtkm::VecFlat<T>::ComponentType>
void TryConvertRuntimeVec()
{
  using BasicArrayType = vtkm::cont::ArrayHandle<T>;
  constexpr vtkm::IdComponent numFlatComponents = vtkm::VecFlat<T>::NUM_COMPONENTS;
  using RuntimeArrayType = vtkm::cont::ArrayHandleRuntimeVec<BasicComponentType>;

  std::cout << "    Get basic array as ArrayHandleRuntimeVec" << std::endl;
  BasicArrayType inputArray;
  inputArray.Allocate(ARRAY_SIZE);
  SetPortal(inputArray.WritePortal());

  vtkm::cont::UnknownArrayHandle unknownWithBasic{ inputArray };
  VTKM_TEST_ASSERT(unknownWithBasic.GetNumberOfComponentsFlat() == numFlatComponents);

  VTKM_TEST_ASSERT(unknownWithBasic.CanConvert<RuntimeArrayType>());
  RuntimeArrayType runtimeArray = unknownWithBasic.AsArrayHandle<RuntimeArrayType>();

  // Hack to convert the array handle to a flat array to make it easy to check the runtime array
  vtkm::cont::ArrayHandle<vtkm::VecFlat<T>> flatInput{ inputArray.GetBuffers() };
  VTKM_TEST_ASSERT(test_equal_ArrayHandles(flatInput, runtimeArray));

  std::cout << "    Get ArrayHandleRuntimeVec as basic array" << std::endl;
  vtkm::cont::UnknownArrayHandle unknownWithRuntimeVec{ runtimeArray };
  VTKM_TEST_ASSERT(unknownWithRuntimeVec.GetNumberOfComponentsFlat() == numFlatComponents);

  VTKM_TEST_ASSERT(unknownWithRuntimeVec.CanConvert<RuntimeArrayType>());
  VTKM_TEST_ASSERT(unknownWithRuntimeVec.CanConvert<BasicArrayType>());
  BasicArrayType outputArray = unknownWithRuntimeVec.AsArrayHandle<BasicArrayType>();
  VTKM_TEST_ASSERT(test_equal_ArrayHandles(inputArray, outputArray));

  std::cout << "    Copy ArrayHandleRuntimeVec to a new instance" << std::endl;
  vtkm::cont::UnknownArrayHandle unknownCopy = unknownWithRuntimeVec.NewInstance();
  VTKM_TEST_ASSERT(unknownWithRuntimeVec.GetNumberOfComponentsFlat() ==
                   unknownCopy.GetNumberOfComponentsFlat());
  vtkm::cont::ArrayCopy(unknownWithRuntimeVec, unknownCopy);
  VTKM_TEST_ASSERT(test_equal_ArrayHandles(inputArray, unknownCopy));

  std::cout << "    Copy ArrayHandleRuntimeVec as basic array" << std::endl;
  unknownCopy = unknownWithRuntimeVec.NewInstanceBasic();
  VTKM_TEST_ASSERT(unknownWithRuntimeVec.GetNumberOfComponentsFlat() ==
                   unknownCopy.GetNumberOfComponentsFlat());
  vtkm::cont::ArrayCopy(unknownWithRuntimeVec, unknownCopy);
  VTKM_TEST_ASSERT(test_equal_ArrayHandles(inputArray, unknownCopy));

  std::cout << "    Copy ArrayHandleRuntimeVec to float array" << std::endl;
  unknownCopy = unknownWithRuntimeVec.NewInstanceFloatBasic();
  VTKM_TEST_ASSERT(unknownWithRuntimeVec.GetNumberOfComponentsFlat() ==
                   unknownCopy.GetNumberOfComponentsFlat());
  vtkm::cont::ArrayCopy(unknownWithRuntimeVec, unknownCopy);
  VTKM_TEST_ASSERT(test_equal_ArrayHandles(inputArray, unknownCopy));
}

void TryConvertRuntimeVec()
{
  std::cout << "  Scalar array." << std::endl;
  TryConvertRuntimeVec<vtkm::FloatDefault>();

  std::cout << "  Equivalent scalar." << std::endl;
  TryConvertRuntimeVec<VTKM_UNUSED_INT_TYPE>();

  std::cout << "  Basic Vec." << std::endl;
  TryConvertRuntimeVec<vtkm::Id3>();

  std::cout << "  Vec of Vecs." << std::endl;
  TryConvertRuntimeVec<vtkm::Vec<vtkm::Vec2f, 3>>();

  std::cout << "  Vec of Vecs of Vecs." << std::endl;
  TryConvertRuntimeVec<vtkm::Vec<vtkm::Vec<vtkm::Id4, 3>, 2>>();

  std::cout << "  Compatible but different C types." << std::endl;
  if (sizeof(int) == sizeof(long))
  {
    TryConvertRuntimeVec<vtkm::Vec<int, 4>, long>();
  }
  else // assuming sizeof(long long) == sizeof(long)
  {
    TryConvertRuntimeVec<vtkm::Vec<long long, 4>, long>();
  }
}

struct DefaultTypeFunctor
{
  template <typename T>
  void operator()(T) const
  {
    TryDefaultType<T>();
  }
};

void TestUnknownArrayHandle()
{
  std::cout << "Try common types with default type lists." << std::endl;
  vtkm::testing::Testing::TryTypes(DefaultTypeFunctor{}, VTKM_DEFAULT_TYPE_LIST{});

  std::cout << "Try exemplar VTK-m types." << std::endl;
  vtkm::testing::Testing::TryTypes(TryBasicVTKmType{});

  std::cout << "Try unusual type." << std::endl;
  TryUnusualType();

  std::cout << "Try AsArrayHandle" << std::endl;
  TryAsArrayHandle();

  std::cout << "Try CastAndCall with fallback" << std::endl;
  TryCastAndCallFallback();

  std::cout << "Try ExtractComponent" << std::endl;
  TryExtractComponent();

  std::cout << "Try setting ArrayHandleCast" << std::endl;
  TrySetCastArray();

  std::cout << "Try setting ArrayHandleMultiplexer" << std::endl;
  TrySetMultiplexerArray();

  std::cout << "Try converting between ArrayHandleRuntimeVec and basic array" << std::endl;
  TryConvertRuntimeVec();
}

} // anonymous namespace

int UnitTestUnknownArrayHandle(int argc, char* argv[])
{
  return vtkm::cont::testing::Testing::Run(TestUnknownArrayHandle, argc, argv);
}
