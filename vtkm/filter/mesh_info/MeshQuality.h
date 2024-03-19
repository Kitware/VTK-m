//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#ifndef vtk_m_filter_mesh_info_MeshQuality_h
#define vtk_m_filter_mesh_info_MeshQuality_h

#include <vtkm/filter/Filter.h>
#include <vtkm/filter/mesh_info/vtkm_filter_mesh_info_export.h>

#include <vtkm/Deprecated.h>

namespace vtkm
{
namespace filter
{
namespace mesh_info
{

enum struct CellMetric
{
  /// @copydoc MeshQualityArea
  Area,
  /// @copydoc MeshQualityAspectGamma
  AspectGamma,
  /// @copydoc MeshQualityAspectRatio
  AspectRatio,
  /// @copydoc MeshQualityCondition
  Condition,
  /// @copydoc MeshQualityDiagonalRatio
  DiagonalRatio,
  /// @copydoc MeshQualityDimension
  Dimension,
  /// @copydoc MeshQualityJacobian
  Jacobian,
  /// @copydoc MeshQualityMaxAngle
  MaxAngle,
  /// @copydoc MeshQualityMaxDiagonal
  MaxDiagonal,
  /// @copydoc MeshQualityMinAngle
  MinAngle,
  /// @copydoc MeshQualityMinDiagonal
  MinDiagonal,
  /// @copydoc MeshQualityOddy
  Oddy,
  /// @copydoc MeshQualityRelativeSizeSquared
  RelativeSizeSquared,
  /// @copydoc MeshQualityScaledJacobian
  ScaledJacobian,
  /// @copydoc MeshQualityShape
  Shape,
  /// @copydoc MeshQualityShapeAndSize
  ShapeAndSize,
  /// @copydoc MeshQualityShear
  Shear,
  /// @copydoc MeshQualitySkew
  Skew,
  /// @copydoc MeshQualityStretch
  Stretch,
  /// @copydoc MeshQualityTaper
  Taper,
  /// @copydoc MeshQualityVolume
  Volume,
  /// @copydoc MeshQualityWarpage
  Warpage,
  None
};

/// @brief Computes the quality of an unstructured cell-based mesh.
///
/// The quality is defined in terms of the summary statistics (frequency, mean, variance,
/// min, max) of metrics computed over the mesh cells. One of several different metrics
/// can be specified for a given cell type, and the mesh can consist of one or more different
/// cell types. The resulting mesh quality is stored as one or more new fields in the output
/// dataset of this filter, with a separate field for each cell type. Each field contains the
/// metric summary statistics for the cell type. Summary statists with all 0 values imply
/// that the specified metric does not support the cell type.
///
class VTKM_FILTER_MESH_INFO_EXPORT MeshQuality : public vtkm::filter::Filter
{
public:
  MeshQuality();

  VTKM_DEPRECATED(2.2, "use default constructor and SetMetric().")
  VTKM_CONT explicit MeshQuality(CellMetric);

  /// @brief Specify the metric to compute on the mesh.
  VTKM_CONT void SetMetric(CellMetric metric);
  /// @copydoc SetMetric
  VTKM_CONT CellMetric GetMetric() const { return this->MyMetric; }

  /// @brief Return a string describing the metric selected.
  VTKM_CONT std::string GetMetricName() const;

private:
  VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input) override;

  CellMetric MyMetric = CellMetric::None;
};
} // namespace mesh_info
} // namespace filter
} // namespace vtkm

#endif // vtk_m_filter_mesh_info_MeshQuality_h
