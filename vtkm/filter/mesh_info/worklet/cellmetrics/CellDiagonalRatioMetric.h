//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2018 UT-Battelle, LLC.
//  Copyright 2018 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================
#ifndef vtk_m_worklet_cellmetrics_CellDiagonalRatioMetric_h
#define vtk_m_worklet_cellmetrics_CellDiagonalRatioMetric_h

/*
* Mesh quality metric functions that compute the diagonal ratio of mesh cells.
* The diagonal ratio of a cell is defined as the length (magnitude) of the longest
* cell diagonal length divided by the length of the shortest cell diagonal length.
** These metric computations are adapted from the VTK implementation of the Verdict library,
* which provides a set of mesh/cell metrics for evaluating the geometric qualities of regions
* of mesh spaces.
** The edge ratio computations for a pyramid cell types is not defined in the
* VTK implementation, but is provided here.
** See: The Verdict Library Reference Manual (for per-cell-type metric formulae)
* See: vtk/ThirdParty/verdict/vtkverdict (for VTK code implementation of this metric)
*/

#include <vtkm/CellShape.h>
#include <vtkm/CellTraits.h>
#include <vtkm/ErrorCode.h>
#include <vtkm/VecTraits.h>
#include <vtkm/VectorAnalysis.h>

#define UNUSED(expr) (void)(expr);

namespace vtkm
{
namespace worklet
{
namespace cellmetrics
{

using FloatType = vtkm::FloatDefault;

template <typename OutType, typename VecType>
VTKM_EXEC inline OutType ComputeDiagonalRatio(const VecType& diagonals)
{
  const vtkm::Id numDiagonals = diagonals.GetNumberOfComponents();

  //Compare diagonal lengths to determine the longest and shortest
  //TODO: Could we use lambda expression here?

  FloatType d0Len = (FloatType)vtkm::MagnitudeSquared(diagonals[0]);
  FloatType currLen, minLen = d0Len, maxLen = d0Len;
  for (int i = 1; i < numDiagonals; i++)
  {
    currLen = (FloatType)vtkm::MagnitudeSquared(diagonals[i]);
    if (currLen < minLen)
      minLen = currLen;
    if (currLen > maxLen)
      maxLen = currLen;
  }

  if (maxLen <= OutType(0.0))
    return vtkm::Infinity<OutType>();

  //Take square root because we only did magnitude squared before
  OutType diagonalRatio = (OutType)vtkm::Sqrt(minLen / maxLen);
  return diagonalRatio;
}

// By default, cells have zero shape unless the shape type template is specialized below.
template <typename OutType, typename PointCoordVecType, typename CellShapeType>
VTKM_EXEC OutType CellDiagonalRatioMetric(const vtkm::IdComponent& numPts,
                                          const PointCoordVecType& pts,
                                          CellShapeType shape,
                                          vtkm::ErrorCode&)
{
  UNUSED(numPts);
  UNUSED(pts);
  UNUSED(shape);
  return OutType(-1.0);
}

// ========================= 2D cells ==================================
// Compute the diagonal ratio of a quadrilateral.
// Formula: Maximum diagonal length divided by minimum diagonal length
// Equals 1 for a unit square
// Full range: [1,FLOAT_MAX]
template <typename OutType, typename PointCoordVecType>
VTKM_EXEC OutType CellDiagonalRatioMetric(const vtkm::IdComponent& numPts,
                                          const PointCoordVecType& pts,
                                          vtkm::CellShapeTagQuad,
                                          vtkm::ErrorCode& ec)
{
  if (numPts != 4)
  {
    ec = vtkm::ErrorCode::InvalidNumberOfPoints;
    return OutType(0.0);
  }

  vtkm::IdComponent numDiagonals = 2; //pts.GetNumberOfComponents();

  //The 2 diagonals of a quadrilateral
  using Diagonal = typename PointCoordVecType::ComponentType;
  const Diagonal QuadDiagonals[2] = { pts[2] - pts[0], pts[3] - pts[1] };

  return vtkm::worklet::cellmetrics::ComputeDiagonalRatio<OutType>(
    vtkm::make_VecC(QuadDiagonals, numDiagonals));
}

// ============================= 3D Volume cells ==================================
// Compute the diagonal ratio of a hexahedron.
// Formula: Maximum diagonal length divided by minimum diagonal length
// Equals 1 for a unit cube
// Acceptable Range: [0.65, 1]
// Normal Range: [0, 1]
// Full range: [1,FLOAT_MAX]
template <typename OutType, typename PointCoordVecType>
VTKM_EXEC OutType CellDiagonalRatioMetric(const vtkm::IdComponent& numPts,
                                          const PointCoordVecType& pts,
                                          vtkm::CellShapeTagHexahedron,
                                          vtkm::ErrorCode& ec)
{
  if (numPts != 8)
  {
    ec = vtkm::ErrorCode::InvalidNumberOfPoints;
    return OutType(0.0);
  }

  vtkm::IdComponent numDiagonals = 4; //pts.GetNumberOfComponents();

  //The 4 diagonals of a hexahedron
  using Diagonal = typename PointCoordVecType::ComponentType;
  const Diagonal HexDiagonals[4] = {
    pts[6] - pts[0], pts[7] - pts[1], pts[4] - pts[2], pts[5] - pts[3]
  };

  return vtkm::worklet::cellmetrics::ComputeDiagonalRatio<OutType>(
    vtkm::make_VecC(HexDiagonals, numDiagonals));
}
} // namespace cellmetrics
} // namespace worklet
} // namespace vtkm
#endif // vtk_m_worklet_cellmetrics_CellDiagonalRatioMetric_h
