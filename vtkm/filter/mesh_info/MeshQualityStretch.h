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
#ifndef vtk_m_filter_mesh_info_MeshQualityStretch_h
#define vtk_m_filter_mesh_info_MeshQualityStretch_h

#include <vtkm/filter/Filter.h>
#include <vtkm/filter/mesh_info/vtkm_filter_mesh_info_export.h>

namespace vtkm
{
namespace filter
{
namespace mesh_info
{

/// @brief Compute the stretch of each cell.
///
/// The stretch of a cell is computed as the ratio of the minimum edge length to the maximum
/// diagonal, normalized for the unit cube. A good quality cell will have a stretch in
/// the range [0.25, 1]. Poorer quality cells can have a stretch as low as 0 although a malformed
/// cell might return a strech of infinity.
///
/// This only produces values for quadrilaterals and hexahedra.
class VTKM_FILTER_MESH_INFO_EXPORT MeshQualityStretch : public vtkm::filter::Filter
{
public:
  MeshQualityStretch();

private:
  vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input) override;
};

} // namespace mesh_info
} // namespace filter
} // namespace vtkm

#endif //vtk_m_filter_mesh_info_MeshQualityStretch_h
