//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_cont_ArrayHandleRecombineVec_h
#define vtk_m_cont_ArrayHandleRecombineVec_h

#include <vtkm/cont/ArrayExtractComponent.h>
#include <vtkm/cont/ArrayHandleMultiplexer.h>
#include <vtkm/cont/ArrayHandleStride.h>
#include <vtkm/cont/ArrayHandleTransform.h>
#include <vtkm/cont/DeviceAdapterTag.h>

#include <vtkm/cont/internal/ArrayRangeComputeUtils.h>

#include <vtkm/internal/ArrayPortalValueReference.h>

namespace vtkm
{
namespace internal
{

// Forward declaration
template <typename SourcePortalType>
class ArrayPortalRecombineVec;

template <typename PortalType>
class RecombineVec
{
  vtkm::VecCConst<PortalType> Portals;
  vtkm::Id Index;

  /// @cond NOPE
  friend vtkm::internal::ArrayPortalRecombineVec<PortalType>;
  /// @endcond

public:
  using ComponentType = typename std::remove_const<typename PortalType::ValueType>::type;

  RecombineVec(const RecombineVec&) = default;

  VTKM_EXEC_CONT RecombineVec(const vtkm::VecCConst<PortalType>& portals, vtkm::Id index)
    : Portals(portals)
    , Index(index)
  {
  }

  VTKM_EXEC_CONT vtkm::IdComponent GetNumberOfComponents() const
  {
    return this->Portals.GetNumberOfComponents();
  }

  VTKM_EXEC_CONT
  vtkm::internal::ArrayPortalValueReference<PortalType> operator[](vtkm::IdComponent cIndex) const
  {
    return vtkm::internal::ArrayPortalValueReference<PortalType>(this->Portals[cIndex],
                                                                 this->Index);
  }

  template <typename T, vtkm::IdComponent DestSize>
  VTKM_EXEC_CONT void CopyInto(vtkm::Vec<T, DestSize>& dest) const
  {
    vtkm::IdComponent numComponents = vtkm::Min(DestSize, this->GetNumberOfComponents());
    for (vtkm::IdComponent cIndex = 0; cIndex < numComponents; ++cIndex)
    {
      dest[cIndex] = this->Portals[cIndex].Get(this->Index);
    }
    // Clear out any components not held by this dynamic Vec-like
    for (vtkm::IdComponent cIndex = numComponents; cIndex < DestSize; ++cIndex)
    {
      dest[cIndex] = vtkm::TypeTraits<T>::ZeroInitialization();
    }
  }

  VTKM_EXEC_CONT vtkm::Id GetIndex() const { return this->Index; }

  VTKM_EXEC_CONT RecombineVec& operator=(const RecombineVec& src)
  {
    if ((&this->Portals[0] != &src.Portals[0]) || (this->Index != src.Index))
    {
      this->DoCopy(src);
    }
    else
    {
      // Copying to myself. Do not need to do anything.
    }
    return *this;
  }

  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator=(const T& src)
  {
    this->DoCopy(src);
    return *this;
  }

  VTKM_EXEC_CONT operator ComponentType() const { return this->Portals[0].Get(this->Index); }

  template <vtkm::IdComponent N>
  VTKM_EXEC_CONT operator vtkm::Vec<ComponentType, N>() const
  {
    vtkm::Vec<ComponentType, N> result;
    this->CopyInto(result);
    return result;
  }

  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator+=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] += VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator-=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] -= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator*=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] *= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator/=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] /= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator%=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] %= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator&=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] &= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator|=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] |= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator^=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] ^= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator>>=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] >>= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }
  template <typename T>
  VTKM_EXEC_CONT RecombineVec& operator<<=(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    VTKM_ASSERT(this->GetNumberOfComponents() == VTraits::GetNumberOfComponents(src));
    for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
    {
      (*this)[cIndex] <<= VTraits::GetComponent(src, cIndex);
    }
    return *this;
  }

private:
  template <typename T>
  VTKM_EXEC_CONT void DoCopy(const T& src)
  {
    using VTraits = vtkm::VecTraits<T>;
    vtkm::IdComponent numComponents = VTraits::GetNumberOfComponents(src);
    if (numComponents > 1)
    {
      if (numComponents > this->GetNumberOfComponents())
      {
        numComponents = this->GetNumberOfComponents();
      }
      for (vtkm::IdComponent cIndex = 0; cIndex < numComponents; ++cIndex)
      {
        this->Portals[cIndex].Set(this->Index,
                                  static_cast<ComponentType>(VTraits::GetComponent(src, cIndex)));
      }
    }
    else
    {
      // Special case when copying from a scalar
      for (vtkm::IdComponent cIndex = 0; cIndex < this->GetNumberOfComponents(); ++cIndex)
      {
        this->Portals[cIndex].Set(this->Index,
                                  static_cast<ComponentType>(VTraits::GetComponent(src, 0)));
      }
    }
  }
};

} // namespace internal

template <typename PortalType>
struct TypeTraits<vtkm::internal::RecombineVec<PortalType>>
{
private:
  using VecType = vtkm::internal::RecombineVec<PortalType>;
  using ComponentType = typename VecType::ComponentType;

public:
  using NumericTag = typename vtkm::TypeTraits<ComponentType>::NumericTag;
  using DimensionalityTag = vtkm::TypeTraitsVectorTag;

  VTKM_EXEC_CONT static vtkm::internal::RecombineVec<PortalType> ZeroInitialization()
  {
    // Return a vec-like of size 0.
    return vtkm::internal::RecombineVec<PortalType>{};
  }
};

template <typename PortalType>
struct VecTraits<vtkm::internal::RecombineVec<PortalType>>
{
  using VecType = vtkm::internal::RecombineVec<PortalType>;
  using ComponentType = typename VecType::ComponentType;
  using BaseComponentType = typename vtkm::VecTraits<ComponentType>::BaseComponentType;
  using HasMultipleComponents = vtkm::VecTraitsTagMultipleComponents;
  using IsSizeStatic = vtkm::VecTraitsTagSizeVariable;

  VTKM_EXEC_CONT static vtkm::IdComponent GetNumberOfComponents(const VecType& vector)
  {
    return vector.GetNumberOfComponents();
  }

  VTKM_EXEC_CONT
  static ComponentType GetComponent(const VecType& vector, vtkm::IdComponent componentIndex)
  {
    return vector[componentIndex];
  }

  VTKM_EXEC_CONT static void SetComponent(const VecType& vector,
                                          vtkm::IdComponent componentIndex,
                                          const ComponentType& component)
  {
    vector[componentIndex] = component;
  }

  template <vtkm::IdComponent destSize>
  VTKM_EXEC_CONT static void CopyInto(const VecType& src, vtkm::Vec<ComponentType, destSize>& dest)
  {
    src.CopyInto(dest);
  }
};

namespace internal
{

template <typename SourcePortalType>
class ArrayPortalRecombineVec
{
  // Note that this ArrayPortal has a pointer to a C array of other portals. We need to
  // make sure that the pointer is valid on the device we are using it on. See the
  // CreateReadPortal and CreateWritePortal in the Storage below to see how that is
  // managed.
  const SourcePortalType* Portals;
  vtkm::IdComponent NumberOfComponents;

public:
  using ValueType = vtkm::internal::RecombineVec<SourcePortalType>;

  ArrayPortalRecombineVec() = default;
  ArrayPortalRecombineVec(const SourcePortalType* portals, vtkm::IdComponent numComponents)
    : Portals(portals)
    , NumberOfComponents(numComponents)
  {
  }

  VTKM_EXEC_CONT vtkm::Id GetNumberOfValues() const { return this->Portals[0].GetNumberOfValues(); }

  VTKM_EXEC_CONT ValueType Get(vtkm::Id index) const
  {
    return ValueType({ this->Portals, this->NumberOfComponents }, index);
  }

  VTKM_EXEC_CONT void Set(vtkm::Id index, const ValueType& value) const
  {
    if ((value.GetIndex() == index) && (value.Portals.GetPointer() == this->Portals))
    {
      // The ValueType is actually a reference back to the portals. If this reference is
      // actually pointing back to the same index, we don't need to do anything.
    }
    else
    {
      this->DoCopy(index, value);
    }
  }

  template <typename T>
  VTKM_EXEC_CONT void Set(vtkm::Id index, const T& value) const
  {
    this->DoCopy(index, value);
  }

private:
  template <typename T>
  VTKM_EXEC_CONT void DoCopy(vtkm::Id index, const T& value) const
  {
    using Traits = vtkm::VecTraits<T>;
    VTKM_ASSERT(Traits::GetNumberOfComponents(value) == this->NumberOfComponents);
    for (vtkm::IdComponent cIndex = 0; cIndex < this->NumberOfComponents; ++cIndex)
    {
      this->Portals[cIndex].Set(index, Traits::GetComponent(value, cIndex));
    }
  }
};

}
} // namespace vtkm::internal

namespace vtkm
{
namespace cont
{

namespace internal
{

struct StorageTagRecombineVec
{
};

namespace detail
{

struct RecombineVecMetaData
{
  mutable std::vector<vtkm::cont::internal::Buffer> PortalBuffers;
  std::vector<std::size_t> ArrayBufferOffsets;

  RecombineVecMetaData() = default;

  RecombineVecMetaData(const RecombineVecMetaData& src) { *this = src; }

  RecombineVecMetaData& operator=(const RecombineVecMetaData& src)
  {
    this->ArrayBufferOffsets = src.ArrayBufferOffsets;

    this->PortalBuffers.clear();
    // Intentionally not copying portals. Portals will be recreated from proper array when requsted.

    return *this;
  }
};

template <typename T>
using RecombinedPortalType = vtkm::internal::ArrayPortalMultiplexer<
  typename vtkm::cont::internal::Storage<T, vtkm::cont::StorageTagStride>::ReadPortalType,
  typename vtkm::cont::internal::Storage<T, vtkm::cont::StorageTagStride>::WritePortalType>;

template <typename T>
using RecombinedValueType = vtkm::internal::RecombineVec<RecombinedPortalType<T>>;

} // namespace detail

template <typename ReadWritePortal>
class Storage<vtkm::internal::RecombineVec<ReadWritePortal>,
              vtkm::cont::internal::StorageTagRecombineVec>
{
  using ComponentType = typename ReadWritePortal::ValueType;
  using SourceStorage = vtkm::cont::internal::Storage<ComponentType, vtkm::cont::StorageTagStride>;
  using ArrayType = vtkm::cont::ArrayHandle<ComponentType, vtkm::cont::StorageTagStride>;

  VTKM_STATIC_ASSERT(
    (std::is_same<ReadWritePortal, detail::RecombinedPortalType<ComponentType>>::value));

  VTKM_CONT static std::vector<vtkm::cont::internal::Buffer> BuffersForComponent(
    const std::vector<vtkm::cont::internal::Buffer>& buffers,
    vtkm::IdComponent componentIndex)
  {
    auto& metaData = buffers[0].GetMetaData<detail::RecombineVecMetaData>();
    std::size_t index = static_cast<std::size_t>(componentIndex);
    return std::vector<vtkm::cont::internal::Buffer>(
      buffers.begin() + metaData.ArrayBufferOffsets[index],
      buffers.begin() + metaData.ArrayBufferOffsets[index + 1]);
  }

public:
  using ReadPortalType = vtkm::internal::ArrayPortalRecombineVec<ReadWritePortal>;
  using WritePortalType = vtkm::internal::ArrayPortalRecombineVec<ReadWritePortal>;

  VTKM_CONT static vtkm::IdComponent GetNumberOfComponents(
    const std::vector<vtkm::cont::internal::Buffer>& buffers)
  {
    return static_cast<vtkm::IdComponent>(
      buffers[0].GetMetaData<detail::RecombineVecMetaData>().ArrayBufferOffsets.size() - 1);
  }

  VTKM_CONT static vtkm::IdComponent GetNumberOfComponentsFlat(
    const std::vector<vtkm::cont::internal::Buffer>& buffers)
  {
    vtkm::IdComponent numComponents = GetNumberOfComponents(buffers);
    vtkm::IdComponent numSubComponents =
      SourceStorage::GetNumberOfComponentsFlat(BuffersForComponent(buffers, 0));
    return numComponents * numSubComponents;
  }

  VTKM_CONT static vtkm::Id GetNumberOfValues(
    const std::vector<vtkm::cont::internal::Buffer>& buffers)
  {
    return SourceStorage::GetNumberOfValues(BuffersForComponent(buffers, 0));
  }

  VTKM_CONT static void ResizeBuffers(vtkm::Id numValues,
                                      const std::vector<vtkm::cont::internal::Buffer>& buffers,
                                      vtkm::CopyFlag preserve,
                                      vtkm::cont::Token& token)
  {
    vtkm::IdComponent numComponents = GetNumberOfComponents(buffers);
    for (vtkm::IdComponent component = 0; component < numComponents; ++component)
    {
      SourceStorage::ResizeBuffers(
        numValues, BuffersForComponent(buffers, component), preserve, token);
    }
  }

  VTKM_CONT static void Fill(const std::vector<vtkm::cont::internal::Buffer>&,
                             const vtkm::internal::RecombineVec<ReadWritePortal>&,
                             vtkm::Id,
                             vtkm::Id,
                             vtkm::cont::Token&)
  {
    throw vtkm::cont::ErrorBadType("Fill not supported for ArrayHandleRecombineVec.");
  }

  VTKM_CONT static ReadPortalType CreateReadPortal(
    const std::vector<vtkm::cont::internal::Buffer>& buffers,
    vtkm::cont::DeviceAdapterId device,
    vtkm::cont::Token& token)
  {
    vtkm::IdComponent numComponents = GetNumberOfComponents(buffers);

    // The array portal needs a runtime-allocated array of portals for each component.
    // We use the vtkm::cont::internal::Buffer object to allow us to allocate memory on the
    // device and copy data there.
    vtkm::cont::internal::Buffer portalBuffer;
    portalBuffer.SetNumberOfBytes(static_cast<vtkm::BufferSizeType>(sizeof(ReadWritePortal)) *
                                    numComponents,
                                  vtkm::CopyFlag::Off,
                                  token);

    // Save a reference of the portal in our metadata.
    // Note that the buffer we create is going to hang around until the ArrayHandle gets
    // destroyed. The buffers are small and should not be a problem unless you create a
    // lot of portals.
    buffers[0].GetMetaData<detail::RecombineVecMetaData>().PortalBuffers.push_back(portalBuffer);

    // Get the control-side memory and fill it with the execution-side portals
    ReadWritePortal* portals =
      reinterpret_cast<ReadWritePortal*>(portalBuffer.WritePointerHost(token));
    for (vtkm::IdComponent cIndex = 0; cIndex < numComponents; ++cIndex)
    {
      portals[cIndex] = ReadWritePortal(
        SourceStorage::CreateReadPortal(BuffersForComponent(buffers, cIndex), device, token));
    }

    // Now get the execution-side memory (portals will be copied as necessary) and create
    // the portal for the appropriate device
    return ReadPortalType(
      reinterpret_cast<const ReadWritePortal*>(portalBuffer.ReadPointerDevice(device, token)),
      numComponents);
  }

  VTKM_CONT static WritePortalType CreateWritePortal(
    const std::vector<vtkm::cont::internal::Buffer>& buffers,
    vtkm::cont::DeviceAdapterId device,
    vtkm::cont::Token& token)
  {
    vtkm::IdComponent numComponents = GetNumberOfComponents(buffers);

    // The array portal needs a runtime-allocated array of portals for each component.
    // We use the vtkm::cont::internal::Buffer object to allow us to allocate memory on the
    // device and copy data there.
    vtkm::cont::internal::Buffer portalBuffer;
    portalBuffer.SetNumberOfBytes(static_cast<vtkm::BufferSizeType>(sizeof(ReadWritePortal)) *
                                    numComponents,
                                  vtkm::CopyFlag::Off,
                                  token);

    // Save a reference of the portal in our metadata.
    // Note that the buffer we create is going to hang around until the ArrayHandle gets
    // destroyed. The buffers are small and should not be a problem unless you create a
    // lot of portals.
    buffers[0].GetMetaData<detail::RecombineVecMetaData>().PortalBuffers.push_back(portalBuffer);

    // Get the control-side memory and fill it with the execution-side portals
    ReadWritePortal* portals =
      reinterpret_cast<ReadWritePortal*>(portalBuffer.WritePointerHost(token));
    for (vtkm::IdComponent cIndex = 0; cIndex < numComponents; ++cIndex)
    {
      portals[cIndex] = ReadWritePortal(
        SourceStorage::CreateWritePortal(BuffersForComponent(buffers, cIndex), device, token));
    }

    // Now get the execution-side memory (portals will be copied as necessary) and create
    // the portal for the appropriate device
    return WritePortalType(
      reinterpret_cast<const ReadWritePortal*>(portalBuffer.ReadPointerDevice(device, token)),
      numComponents);
  }

  VTKM_CONT static ArrayType ArrayForComponent(
    const std::vector<vtkm::cont::internal::Buffer>& buffers,
    vtkm::IdComponent componentIndex)
  {
    return ArrayType(BuffersForComponent(buffers, componentIndex));
  }

  VTKM_CONT static std::vector<vtkm::cont::internal::Buffer> CreateBuffers()
  {
    detail::RecombineVecMetaData metaData;
    metaData.ArrayBufferOffsets.push_back(1);
    return vtkm::cont::internal::CreateBuffers(metaData);
  }

  VTKM_CONT static void AppendComponent(std::vector<vtkm::cont::internal::Buffer>& buffers,
                                        const ArrayType& array)
  {
    // Add buffers of new array to our list of buffers.
    buffers.insert(buffers.end(), array.GetBuffers().begin(), array.GetBuffers().end());
    // Update metadata for new offset to end.
    buffers[0].GetMetaData<detail::RecombineVecMetaData>().ArrayBufferOffsets.push_back(
      buffers.size());
  }
};

} // namespace internal

/// @brief A grouping of `ArrayHandleStride`s into an `ArrayHandle` of `vtkm::Vec`s.
///
/// The main intention of `ArrayHandleStride` is to pull out a component of an
/// `ArrayHandle` without knowing there `ArrayHandle`'s storage or `vtkm::Vec` shape.
/// However, usually you want to do an operation on all the components together.
/// `ArrayHandleRecombineVec` implements the functionality to easily take a
/// group of extracted components and treat them as a single `ArrayHandle` of
/// `vtkm::Vec` values.
///
/// Note that caution should be used with `ArrayHandleRecombineVec` because the
/// size of the `vtkm::Vec` values is not known at compile time. Thus, the value
/// type of this array is forced to a special `RecombineVec` class that can cause
/// surprises if treated as a `vtkm::Vec`. In particular, the static `NUM_COMPONENTS`
/// expression does not exist. Furthermore, new variables of type `RecombineVec`
/// cannot be created. This means that simple operators like `+` will not work
/// because they require an intermediate object to be created. (Equal operators
/// like `+=` do work because they are given an existing variable to place the
/// output.)
///
template <typename ComponentType>
class ArrayHandleRecombineVec
  : public vtkm::cont::ArrayHandle<internal::detail::RecombinedValueType<ComponentType>,
                                   vtkm::cont::internal::StorageTagRecombineVec>
{
public:
  VTKM_ARRAY_HANDLE_SUBCLASS(
    ArrayHandleRecombineVec,
    (ArrayHandleRecombineVec<ComponentType>),
    (vtkm::cont::ArrayHandle<internal::detail::RecombinedValueType<ComponentType>,
                             vtkm::cont::internal::StorageTagRecombineVec>));

  /// @brief Return the number of components in each value of the array.
  ///
  /// This is also equal to the number of component arrays referenced by this
  /// fancy array.
  ///
  /// `ArrayHandleRecombineVec` always stores flat Vec values. As such, this number
  /// of components is the same as the number of base components.
  vtkm::IdComponent GetNumberOfComponents() const
  {
    return StorageType::GetNumberOfComponents(this->GetBuffers());
  }

  /// @brief Get the array storing the values for a particular component.
  ///
  /// The returned array is a `vtkm::cont::ArrayHandleStride`. It is possible
  /// that the returned arrays from different components reference the same area
  /// of physical memory (usually referencing values interleaved with each other).
  vtkm::cont::ArrayHandleStride<ComponentType> GetComponentArray(
    vtkm::IdComponent componentIndex) const
  {
    return StorageType::ArrayForComponent(this->GetBuffers(), componentIndex);
  }

  /// @brief Add a component array.
  ///
  /// `AppendComponentArray()` provides an easy way to build an `ArrayHandleRecombineVec`
  /// by iteratively adding the component arrays.
  void AppendComponentArray(
    const vtkm::cont::ArrayHandle<ComponentType, vtkm::cont::StorageTagStride>& array)
  {
    std::vector<vtkm::cont::internal::Buffer> buffers = this->GetBuffers();
    StorageType::AppendComponent(buffers, array);
    this->SetBuffers(std::move(buffers));
  }
};

namespace internal
{

template <>
struct ArrayExtractComponentImpl<vtkm::cont::internal::StorageTagRecombineVec>
{
  template <typename RecombineVec>
  vtkm::cont::ArrayHandleStride<
    typename vtkm::VecFlat<typename RecombineVec::ComponentType>::ComponentType>
  operator()(
    const vtkm::cont::ArrayHandle<RecombineVec, vtkm::cont::internal::StorageTagRecombineVec>& src,
    vtkm::IdComponent componentIndex,
    vtkm::CopyFlag allowCopy) const
  {
    using ComponentType = typename RecombineVec::ComponentType;
    vtkm::cont::ArrayHandleRecombineVec<ComponentType> array(src);
    constexpr vtkm::IdComponent subComponents = vtkm::VecFlat<ComponentType>::NUM_COMPONENTS;
    return vtkm::cont::ArrayExtractComponent(
      array.GetComponentArray(componentIndex / subComponents),
      componentIndex % subComponents,
      allowCopy);
  }
};

//-------------------------------------------------------------------------------------------------
template <typename S>
struct ArrayRangeComputeImpl;

template <typename S>
struct ArrayRangeComputeMagnitudeImpl;

template <typename T, typename S>
inline vtkm::cont::ArrayHandle<vtkm::Range> ArrayRangeComputeImplCaller(
  const vtkm::cont::ArrayHandle<T, S>& input,
  const vtkm::cont::ArrayHandle<vtkm::UInt8>& maskArray,
  bool computeFiniteRange,
  vtkm::cont::DeviceAdapterId device)
{
  return vtkm::cont::internal::ArrayRangeComputeImpl<S>{}(
    input, maskArray, computeFiniteRange, device);
}

template <typename T, typename S>
inline vtkm::Range ArrayRangeComputeMagnitudeImplCaller(
  const vtkm::cont::ArrayHandle<T, S>& input,
  const vtkm::cont::ArrayHandle<vtkm::UInt8>& maskArray,
  bool computeFiniteRange,
  vtkm::cont::DeviceAdapterId device)
{
  return vtkm::cont::internal::ArrayRangeComputeMagnitudeImpl<S>{}(
    input, maskArray, computeFiniteRange, device);
}

template <>
struct VTKM_CONT_EXPORT ArrayRangeComputeImpl<vtkm::cont::internal::StorageTagRecombineVec>
{
  template <typename RecombineVecType>
  VTKM_CONT vtkm::cont::ArrayHandle<vtkm::Range> operator()(
    const vtkm::cont::ArrayHandle<RecombineVecType, vtkm::cont::internal::StorageTagRecombineVec>&
      input_,
    const vtkm::cont::ArrayHandle<vtkm::UInt8>& maskArray,
    bool computeFiniteRange,
    vtkm::cont::DeviceAdapterId device) const
  {
    auto input =
      static_cast<vtkm::cont::ArrayHandleRecombineVec<typename RecombineVecType::ComponentType>>(
        input_);

    vtkm::cont::ArrayHandle<vtkm::Range> result;
    result.Allocate(input.GetNumberOfComponents());

    if (input.GetNumberOfValues() < 1)
    {
      result.Fill(vtkm::Range{});
      return result;
    }

    auto resultPortal = result.WritePortal();
    for (vtkm::IdComponent i = 0; i < input.GetNumberOfComponents(); ++i)
    {
      auto rangeAH = ArrayRangeComputeImplCaller(
        input.GetComponentArray(i), maskArray, computeFiniteRange, device);
      resultPortal.Set(i, rangeAH.ReadPortal().Get(0));
    }

    return result;
  }
};

template <typename ArrayHandleType>
struct ArrayValueIsNested;

template <typename RecombineVecType>
struct ArrayValueIsNested<
  vtkm::cont::ArrayHandle<RecombineVecType, vtkm::cont::internal::StorageTagRecombineVec>>
{
  static constexpr bool Value = false;
};

template <>
struct VTKM_CONT_EXPORT ArrayRangeComputeMagnitudeImpl<vtkm::cont::internal::StorageTagRecombineVec>
{
  template <typename RecombineVecType>
  VTKM_CONT vtkm::Range operator()(
    const vtkm::cont::ArrayHandle<RecombineVecType, vtkm::cont::internal::StorageTagRecombineVec>&
      input_,
    const vtkm::cont::ArrayHandle<vtkm::UInt8>& maskArray,
    bool computeFiniteRange,
    vtkm::cont::DeviceAdapterId device) const
  {
    auto input =
      static_cast<vtkm::cont::ArrayHandleRecombineVec<typename RecombineVecType::ComponentType>>(
        input_);

    if (input.GetNumberOfValues() < 1)
    {
      return vtkm::Range{};
    }
    if (input.GetNumberOfComponents() == 1)
    {
      return ArrayRangeComputeMagnitudeImplCaller(
        input.GetComponentArray(0), maskArray, computeFiniteRange, device);
    }

    return ArrayRangeComputeMagnitudeGeneric(input_, maskArray, computeFiniteRange, device);
  }
};

} // namespace internal

}
} // namespace vtkm::cont

#endif //vtk_m_cont_ArrayHandleRecombineVec_h
