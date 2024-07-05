//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_cont_ArrayHandleUniformPointCoordinates_h
#define vtk_m_cont_ArrayHandleUniformPointCoordinates_h

#include <vtkm/Range.h>
#include <vtkm/cont/ArrayHandleImplicit.h>
#include <vtkm/internal/ArrayPortalUniformPointCoordinates.h>

namespace vtkm
{
namespace cont
{

struct VTKM_ALWAYS_EXPORT StorageTagUniformPoints
{
};

namespace internal
{

using StorageTagUniformPointsSuperclass =
  vtkm::cont::StorageTagImplicit<vtkm::internal::ArrayPortalUniformPointCoordinates>;

template <>
struct Storage<vtkm::Vec3f, vtkm::cont::StorageTagUniformPoints>
  : Storage<vtkm::Vec3f, StorageTagUniformPointsSuperclass>
{
};

} // namespace internal

/// ArrayHandleUniformPointCoordinates is a specialization of ArrayHandle. It
/// contains the information necessary to compute the point coordinates in a
/// uniform orthogonal grid (extent, origin, and spacing) and implicitly
/// computes these coordinates in its array portal.
///
class VTKM_CONT_EXPORT ArrayHandleUniformPointCoordinates
  : public vtkm::cont::ArrayHandle<vtkm::Vec3f, vtkm::cont::StorageTagUniformPoints>
{
public:
  VTKM_ARRAY_HANDLE_SUBCLASS_NT(
    ArrayHandleUniformPointCoordinates,
    (vtkm::cont::ArrayHandle<vtkm::Vec3f, vtkm::cont::StorageTagUniformPoints>));

  /// Create an `ArrayHandleUniformPointCoordinates` with the given specifications.
  VTKM_CONT
  ArrayHandleUniformPointCoordinates(vtkm::Id3 dimensions,
                                     ValueType origin = ValueType(0.0f, 0.0f, 0.0f),
                                     ValueType spacing = ValueType(1.0f, 1.0f, 1.0f));

  // Implemented so that it is defined exclusively in the control environment.
  // If there is a separate device for the execution environment (for example,
  // with CUDA), then the automatically generated destructor could be
  // created for all devices, and it would not be valid for all devices.
  ~ArrayHandleUniformPointCoordinates();

  /// Get the number of points of the uniform grid in the x, y, and z directions.
  VTKM_CONT vtkm::Id3 GetDimensions() const;
  /// Get the coordinates of the "lower-left" cornder of the mesh.
  VTKM_CONT vtkm::Vec3f GetOrigin() const;
  /// Get the spacing between points of the grid in the x, y, and z directions.
  VTKM_CONT vtkm::Vec3f GetSpacing() const;
};

template <typename T>
class ArrayHandleStride;

namespace internal
{

template <typename S>
struct ArrayExtractComponentImpl;

template <>
struct VTKM_CONT_EXPORT ArrayExtractComponentImpl<vtkm::cont::StorageTagUniformPoints>
{
  vtkm::cont::ArrayHandleStride<vtkm::FloatDefault> operator()(
    const vtkm::cont::ArrayHandleUniformPointCoordinates& src,
    vtkm::IdComponent componentIndex,
    vtkm::CopyFlag allowCopy) const;
};

template <typename S>
struct ArrayRangeComputeImpl;

template <>
struct VTKM_CONT_EXPORT ArrayRangeComputeImpl<vtkm::cont::StorageTagUniformPoints>
{
  VTKM_CONT vtkm::cont::ArrayHandle<vtkm::Range> operator()(
    const vtkm::cont::ArrayHandleUniformPointCoordinates& input,
    const vtkm::cont::ArrayHandle<vtkm::UInt8>& maskArray,
    bool computeFiniteRange,
    vtkm::cont::DeviceAdapterId device) const;
};

} // namespace internal

}
} // namespace vtkm::cont

//=============================================================================
// Specializations of serialization related classes
/// @cond SERIALIZATION
namespace vtkm
{
namespace cont
{

template <>
struct SerializableTypeString<vtkm::cont::ArrayHandleUniformPointCoordinates>
{
  static VTKM_CONT std::string Get() { return "AH_UniformPointCoordinates"; }
};

template <>
struct SerializableTypeString<
  vtkm::cont::ArrayHandle<vtkm::Vec3f, vtkm::cont::StorageTagUniformPoints>>
  : SerializableTypeString<vtkm::cont::ArrayHandleUniformPointCoordinates>
{
};
}
} // vtkm::cont

namespace mangled_diy_namespace
{

template <>
struct Serialization<vtkm::cont::ArrayHandleUniformPointCoordinates>
{
private:
  using Type = vtkm::cont::ArrayHandleUniformPointCoordinates;
  using BaseType = vtkm::cont::ArrayHandle<typename Type::ValueType, typename Type::StorageTag>;

public:
  static VTKM_CONT void save(BinaryBuffer& bb, const BaseType& obj)
  {
    auto portal = obj.ReadPortal();
    vtkmdiy::save(bb, portal.GetDimensions());
    vtkmdiy::save(bb, portal.GetOrigin());
    vtkmdiy::save(bb, portal.GetSpacing());
  }

  static VTKM_CONT void load(BinaryBuffer& bb, BaseType& obj)
  {
    vtkm::Id3 dims;
    typename BaseType::ValueType origin, spacing;

    vtkmdiy::load(bb, dims);
    vtkmdiy::load(bb, origin);
    vtkmdiy::load(bb, spacing);

    obj = vtkm::cont::ArrayHandleUniformPointCoordinates(dims, origin, spacing);
  }
};

template <>
struct Serialization<vtkm::cont::ArrayHandle<vtkm::Vec3f, vtkm::cont::StorageTagUniformPoints>>
  : Serialization<vtkm::cont::ArrayHandleUniformPointCoordinates>
{
};

} // diy
/// @endcond SERIALIZATION

#endif //vtk_+m_cont_ArrayHandleUniformPointCoordinates_h
