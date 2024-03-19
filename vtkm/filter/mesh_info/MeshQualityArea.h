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
#ifndef vtk_m_filter_mesh_info_MeshQualityArea_h
#define vtk_m_filter_mesh_info_MeshQualityArea_h

#include <vtkm/filter/Filter.h>
#include <vtkm/filter/mesh_info/vtkm_filter_mesh_info_export.h>

namespace vtkm
{
namespace filter
{
namespace mesh_info
{

/// @brief Compute the area of each cell.
///
/// This only produces values for triangles and quadrilaterals.
class VTKM_FILTER_MESH_INFO_EXPORT MeshQualityArea : public vtkm::filter::Filter
{
public:
  MeshQualityArea();

  /// @brief Computes the area of all polygonal cells and returns the total area.
  vtkm::Float64 ComputeTotalArea(const vtkm::cont::DataSet& input);

  /// @brief Computes the average area of cells.
  ///
  /// This method first computes the total area of all cells and then divides that by the
  /// number of cells in the dataset.
  vtkm::Float64 ComputeAverageArea(const vtkm::cont::DataSet& input);

private:
  vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input) override;
};

} // namespace mesh_info
} // namespace filter
} // namespace vtkm

#endif //vtk_m_filter_mesh_info_MeshQualityArea_h
