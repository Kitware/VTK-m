//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#ifndef vtk_m_filter_field_transform_Warp_h
#define vtk_m_filter_field_transform_Warp_h

#include <vtkm/filter/Filter.h>
#include <vtkm/filter/field_transform/vtkm_filter_field_transform_export.h>

namespace vtkm
{
namespace filter
{
namespace field_transform
{

/// \brief Modify points by moving points along scaled direction vectors.
///
/// This filter displaces the point coordinates of a dataset either in the direction
/// of a direction vector field or in a constant direction.
///
/// The filter starts with a set of point coordinates or other vectors. By default these vectors
/// are the coordinate system, but they can be changed by modifying active field 0. These vectors
/// are then displaced by a set of vectors. This is done by selecting a field of directions, a
/// field of scales, and an additional scale factor. The directions are multiplied by the scale
/// field and the scale factor, and this displacement is added to the vector.
///
/// It is common to wish to warp in a constant direction by a scaled amount. To support
/// this so called "WarpScalar", the `Warp` filter allows you to specify a constant
/// direction direction with the `SetConstantDirection()` method. When this is set,
/// no direction field is retrieved. By default `Warp` uses (0, 0, 1) as the direction
/// direction.
///
/// It is also common to wish to simply apply a vector direction field (with a possible
/// constant scale). To support this so called "WarpVector", the `Warp` filter allows you
/// to ignore the scale field with the `SetUseScaleField()` method. When this is unset,
/// no scale field is retrieved. Calling `SetScaleField()` turns on the `UseScaleField`
/// flag. By default, `Warp` uses will not use the scale field unless specified.
///
/// The main use case for `Warp` is to adjust the spatial location and shape
/// of objects in 3D space. This filter will operate on the `vtkm::cont::CoordinateSystem`
/// of the input data unless a different active field is specified. Likewise,
/// this filter will save its results as the first coordinate system in the output
/// unless `SetChangeCoordinateSystem()` is set to say otherwise.
///
class VTKM_FILTER_FIELD_TRANSFORM_EXPORT Warp : public vtkm::filter::Filter
{
public:
  VTKM_CONT Warp();

  /// @brief Specify a field to use as the directions.
  ///
  /// The directions, when not set to use constant directions, are set as active field index 1.
  VTKM_CONT void SetDirectionField(const std::string& name)
  {
    this->UseConstantDirection = false;
    this->SetActiveField(1, name, vtkm::cont::Field::Association::Points);
  }
  /// @copydoc SetDirectionField
  VTKM_CONT std::string GetDirectionFieldName() const { return this->GetActiveFieldName(1); }

  /// @brief Specify a constant value to use as the directions.
  ///
  /// This will provide a (constant) direction of the direction, and the direction field
  /// will be ignored.
  VTKM_CONT void SetConstantDirection(const vtkm::Vec3f& direction)
  {
    this->UseConstantDirection = true;
    this->ConstantDirection = direction;
  }
  /// @copydoc SetConstantDirection
  VTKM_CONT const vtkm::Vec3f& GetConstantDirection() const { return this->ConstantDirection; }

  /// @brief Specifies whether a direction field or a constant direction direction is used.
  ///
  /// When true, the constant direction direction is used. When false, the direction field (active
  /// field index 1) is used.
  VTKM_CONT void SetUseConstantDirection(bool flag) { this->UseConstantDirection = flag; }
  /// @copydoc SetUseConstantDirection
  VTKM_CONT bool GetUseConstantDirection() const { return this->UseConstantDirection; }

  /// @brief Specify a field to use to scale the directions.
  ///
  /// The scale factor field scales the size of the direction.
  /// The scale factor, when not set to use a constant factor, is set as active field index 2.
  VTKM_CONT void SetScaleField(const std::string& name)
  {
    this->UseScaleField = true;
    this->SetActiveField(2, name, vtkm::cont::Field::Association::Points);
  }
  /// @copydoc SetScaleField
  VTKM_CONT std::string GetScaleFieldName() const { return this->GetActiveFieldName(2); }

  /// @brief Specifies whether a scale factor field is used.
  ///
  /// When true, a scale factor field the constant scale factor is used. When false, the scale factor field (active
  /// field index 2) is used.
  VTKM_CONT void SetUseScaleField(bool flag) { this->UseScaleField = flag; }
  /// @copydoc SetUseScaleField
  VTKM_CONT bool GetUseScaleField() const { return this->UseScaleField; }

  /// @brief Specifies an additional scale factor to scale the displacements.
  ///
  /// When using a non-constant scale field, it is possible that the scale field is
  /// of the wrong units and needs to be rescaled. This scale factor is multiplied to the
  /// direction and scale to re-adjust the overall scale.
  VTKM_CONT void SetScaleFactor(vtkm::FloatDefault scale) { this->ScaleFactor = scale; }
  /// @copydoc SetScaleFactor
  VTKM_CONT vtkm::FloatDefault GetScaleFactor() const { return this->ScaleFactor; }

  /// @brief Specify whether the result should become the coordinate system of the output.
  ///
  /// When this flag is on (the default) the first coordinate system in the output
  /// `vtkm::cont::DataSet` is set to the transformed point coordinates.
  void SetChangeCoordinateSystem(bool flag) { this->ChangeCoordinateSystem = flag; }
  /// @copydoc SetChangeCoordinateSystem
  bool GetChangeCoordinateSystem() const { return this->ChangeCoordinateSystem; }

protected:
  VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input) override;

private:
  vtkm::Vec3f ConstantDirection = { 0, 0, 1 };
  vtkm::FloatDefault ScaleFactor = 1;
  bool UseConstantDirection = true;
  bool UseScaleField = false;
  bool ChangeCoordinateSystem = true;
};

} // namespace field_transform
} // namespace filter
} // namespace vtkm

#endif // vtk_m_filter_field_transform_Warp_h
