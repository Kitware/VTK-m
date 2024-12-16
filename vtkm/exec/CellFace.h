//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_exec_CellFace_h
#define vtk_m_exec_CellFace_h

#include <vtkm/CellShape.h>
#include <vtkm/ErrorCode.h>
#include <vtkm/Types.h>
#include <vtkm/exec/FunctorBase.h>

namespace vtkm
{
namespace exec
{

namespace detail
{

class CellFaceTables
{
public:
  static constexpr vtkm::Int32 MAX_FACE_SIZE = 4;
  static constexpr vtkm::Int32 MAX_NUM_FACES = 6;

  VTKM_EXEC vtkm::Int32 NumFaces(vtkm::Int32 cellShapeId) const
  {
    VTKM_STATIC_CONSTEXPR_ARRAY vtkm::Int32 numFaces[vtkm::NUMBER_OF_CELL_SHAPES] = {
      // NumFaces
      0, //  0: CELL_SHAPE_EMPTY
      0, //  1: CELL_SHAPE_VERTEX
      0, //  2: Unused
      0, //  3: CELL_SHAPE_LINE
      0, //  4: CELL_SHAPE_POLY_LINE
      0, //  5: CELL_SHAPE_TRIANGLE
      0, //  6: Unused
      0, //  7: CELL_SHAPE_POLYGON
      0, //  8: Unused
      0, //  9: CELL_SHAPE_QUAD
      4, // 10: CELL_SHAPE_TETRA
      0, // 11: Unused
      6, // 12: CELL_SHAPE_HEXAHEDRON
      5, // 13: CELL_SHAPE_WEDGE
      5  // 14: CELL_SHAPE_PYRAMID
    };
    return numFaces[cellShapeId];
  }

  VTKM_EXEC vtkm::Int32 NumPointsInFace(vtkm::Int32 cellShapeId, vtkm::Int32 faceIndex) const
  {
    VTKM_STATIC_CONSTEXPR_ARRAY vtkm::Int32
      numPointsInFace[vtkm::NUMBER_OF_CELL_SHAPES][MAX_NUM_FACES] = {
        // NumPointsInFace
        { -1, -1, -1, -1, -1, -1 }, //  0: CELL_SHAPE_EMPTY
        { -1, -1, -1, -1, -1, -1 }, //  1: CELL_SHAPE_VERTEX
        { -1, -1, -1, -1, -1, -1 }, //  2: Unused
        { -1, -1, -1, -1, -1, -1 }, //  3: CELL_SHAPE_LINE
        { -1, -1, -1, -1, -1, -1 }, //  4: CELL_SHAPE_POLY_LINE
        { -1, -1, -1, -1, -1, -1 }, //  5: CELL_SHAPE_TRIANGLE
        { -1, -1, -1, -1, -1, -1 }, //  6: Unused
        { -1, -1, -1, -1, -1, -1 }, //  7: CELL_SHAPE_POLYGON
        { -1, -1, -1, -1, -1, -1 }, //  8: Unused
        { -1, -1, -1, -1, -1, -1 }, //  9: CELL_SHAPE_QUAD
        { 3, 3, 3, 3, -1, -1 },     // 10: CELL_SHAPE_TETRA
        { -1, -1, -1, -1, -1, -1 }, // 11: Unused
        { 4, 4, 4, 4, 4, 4 },       // 12: CELL_SHAPE_HEXAHEDRON
        { 3, 3, 4, 4, 4, -1 },      // 13: CELL_SHAPE_WEDGE
        { 4, 3, 3, 3, 3, -1 }       // 14: CELL_SHAPE_PYRAMID
      };
    return numPointsInFace[cellShapeId][faceIndex];
  }

  VTKM_EXEC vtkm::Int32 PointsInFace(vtkm::Int32 cellShapeId,
                                     vtkm::Int32 faceIndex,
                                     vtkm::Int32 localPointIndex) const
  {
    // clang-format off
    VTKM_STATIC_CONSTEXPR_ARRAY vtkm::Int32 pointsInFace[vtkm::NUMBER_OF_CELL_SHAPES][MAX_NUM_FACES]
                                                  [MAX_FACE_SIZE] =
    {
      // PointsInFace
      //  0: CELL_SHAPE_EMPTY
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  1: CELL_SHAPE_VERTEX
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  2: Unused
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  3: CELL_SHAPE_LINE
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  4: CELL_SHAPE_POLY_LINE
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  5: CELL_SHAPE_TRIANGLE
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  6: Unused
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  7: CELL_SHAPE_POLYGON
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  8: Unused
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      //  9: CELL_SHAPE_QUAD
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      // 10: CELL_SHAPE_TETRA
      { { 0, 1, 3, -1 }, { 1, 2, 3, -1 }, { 2, 0, 3, -1 },
        { 0, 2, 1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      // 11: Unused
      { { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
        { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 } },
      // 12: CELL_SHAPE_HEXAHEDRON
      { { 0, 4, 7, 3 }, { 1, 2, 6, 5 }, { 0, 1, 5, 4 },
        { 3, 7, 6, 2 }, { 0, 3, 2, 1 }, { 4, 5, 6, 7 } },
      // 13: CELL_SHAPE_WEDGE
      { { 0, 1, 2, -1 }, { 3, 5, 4, -1 }, { 0, 3, 4, 1 },
        { 1, 4, 5, 2 },  { 2, 5, 3, 0 }, { -1, -1, -1, -1 } },
      // 14: CELL_SHAPE_PYRAMID
      { { 0, 3, 2, 1 }, { 0, 1, 4, -1 }, { 1, 2, 4, -1 },
        { 2, 3, 4, -1 }, { 3, 0, 4, -1 },{ -1, -1, -1, -1 } }
        // clang-format on
      };
    return pointsInFace[cellShapeId][faceIndex][localPointIndex];
  }
};

} // namespace detail

/// @brief Get the number of faces in a cell.
///
/// @param[in]  shape A tag of type `CellShapeTag*` to identify the shape of the cell.
///     This method is overloaded for different shape types.
/// @param[out] result A reference to return the number of faces.
template <typename CellShapeTag>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceNumberOfFaces(CellShapeTag shape,
                                                              vtkm::IdComponent& result)
{
  (void)shape; //C4100 false positive workaround
  detail::CellFaceTables table;
  result = table.NumFaces(shape.Id);
  return vtkm::ErrorCode::Success;
}

/// @brief Get the number of points in a face.
///
/// Given a local index to the face and a shape of the cell, this method returns the
/// number of points in that particular face.
///
/// @param[in]  faceIndex The index of the face within the cell.
/// @param[in]  shape A tag of type `CellShapeTag*` to identify the shape of the cell.
///     This method is overloaded for different shape types.
/// @param[out] result A reference to return the number of points in the selected face.
template <typename CellShapeTag>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceNumberOfPoints(vtkm::IdComponent faceIndex,
                                                               CellShapeTag shape,
                                                               vtkm::IdComponent& result)
{
  if ((faceIndex < 0) || (faceIndex >= detail::CellFaceTables::MAX_NUM_FACES))
  {
    result = -1;
    return vtkm::ErrorCode::InvalidFaceId;
  }

  vtkm::IdComponent numFaces;
  VTKM_RETURN_ON_ERROR(vtkm::exec::CellFaceNumberOfFaces(shape, numFaces));
  if (faceIndex >= numFaces)
  {
    result = -1;
    return vtkm::ErrorCode::InvalidFaceId;
  }
  detail::CellFaceTables table;
  result = table.NumPointsInFace(shape.Id, faceIndex);
  return vtkm::ErrorCode::Success;
}

/// @brief Get the shape of a face.
///
/// Given a local index to the face and a shape of the cell, this method returns the
/// identifier for the shape of that face.
/// Faces are always polygons, so it is valid to just to treat the face as a
/// `CELL_SHAPE_POLYGON`. However, the face will be checked to see if it can be
/// further specialized to `CELL_SHAPE_TRIANGLE` or `CELL_SHAPE_QUAD`.
///
/// @param[in]  faceIndex The index of the face within the cell.
/// @param[in]  shape A tag of type `CellShapeTag*` to identify the shape of the cell.
///     This method is overloaded for different shape types.
/// @param[out] result A reference to return the number of points in the selected face.
template <typename CellShapeTag>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceShape(vtkm::IdComponent faceIndex,
                                                      CellShapeTag shape,
                                                      vtkm::UInt8& result)
{

  if ((faceIndex < 0) || (faceIndex >= detail::CellFaceTables::MAX_NUM_FACES))
  {
    result = vtkm::CELL_SHAPE_EMPTY;
    return vtkm::ErrorCode::InvalidFaceId;
  }

  vtkm::IdComponent numFacePoints;
  VTKM_RETURN_ON_ERROR(CellFaceNumberOfPoints(faceIndex, shape, numFacePoints));
  switch (numFacePoints)
  {
    case 3:
      result = vtkm::CELL_SHAPE_TRIANGLE;
      break;
    case 4:
      result = vtkm::CELL_SHAPE_QUAD;
      break;
    default:
      result = vtkm::CELL_SHAPE_POLYGON;
      break;
  }
  return vtkm::ErrorCode::Success;
}

/// Given the index for a face of a cell and one of the points on that face, this
/// function returns the point index for the cell.
/// To get the point indices relative to the data set, the returned index should be used
/// to reference a `PointIndices` list.
///
/// @param[in]  pointIndex The index of the edge within the cell.
/// @param[in]  faceIndex The index of the point on the face.
/// @param[in]  shape A tag of type `CellShapeTag*` to identify the shape of the cell.
///     This method is overloaded for different shape types.
/// @param[out] result Reference to put the index of the point relative to the cell
///     (between 0 and the number of points in the cell).
template <typename CellShapeTag>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceLocalIndex(vtkm::IdComponent pointIndex,
                                                           vtkm::IdComponent faceIndex,
                                                           CellShapeTag shape,
                                                           vtkm::IdComponent& result)
{
  vtkm::IdComponent numPointsInFace;
  result = -1;
  VTKM_RETURN_ON_ERROR(vtkm::exec::CellFaceNumberOfPoints(faceIndex, shape, numPointsInFace));
  if (numPointsInFace < 1)
  {
    // An invalid face. We should already have gotten an error from
    // CellFaceNumberOfPoints.
    return vtkm::ErrorCode::InvalidFaceId;
  }

  detail::CellFaceTables table;
  result = table.PointsInFace(shape.Id, faceIndex, pointIndex);
  return vtkm::ErrorCode::Success;
}

/// @brief Returns a canonical identifier for a cell face
///
/// Given information about a cell face and the global point indices for that cell, returns a
/// `vtkm::Id3` that contains values that are unique to that face. The values for two faces will be
/// the same if and only if the faces contain the same points.
///
/// Note that this property is only true if the mesh is conforming. That is, any two neighboring
/// cells that share a face have the same points on that face. This preculdes 2 faces sharing more
/// than a single point or single edge.
///
template <typename CellShapeTag, typename GlobalPointIndicesVecType>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceCanonicalId(
  vtkm::IdComponent faceIndex,
  CellShapeTag shape,
  const GlobalPointIndicesVecType& globalPointIndicesVec,
  vtkm::Id3& result)
{
  vtkm::IdComponent numPointsInFace;
  result = { -1 };
  VTKM_RETURN_ON_ERROR(vtkm::exec::CellFaceNumberOfPoints(faceIndex, shape, numPointsInFace));
  if (numPointsInFace < 1)
  {
    // An invalid face. We should already have gotten an error from
    // CellFaceNumberOfPoints.
    return vtkm::ErrorCode::InvalidFaceId;
  }

  detail::CellFaceTables table;
  //Sort the first 3 face points/nodes in ascending order
  result = vtkm::Id3(globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 0)],
                     globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 1)],
                     globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 2)]);
  vtkm::Id temp;
  if (result[0] > result[2])
  {
    temp = result[0];
    result[0] = result[2];
    result[2] = temp;
  }
  if (result[0] > result[1])
  {
    temp = result[0];
    result[0] = result[1];
    result[1] = temp;
  }
  if (result[1] > result[2])
  {
    temp = result[1];
    result[1] = result[2];
    result[2] = temp;
  }

  // Check the rest of the points to see if they are in the lowest 3
  for (vtkm::IdComponent pointIndex = 3; pointIndex < numPointsInFace; pointIndex++)
  {
    vtkm::Id nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, pointIndex)];
    if (nextPoint < result[2])
    {
      if (nextPoint < result[1])
      {
        result[2] = result[1];
        if (nextPoint < result[0])
        {
          result[1] = result[0];
          result[0] = nextPoint;
        }
        else // nextPoint > P0, nextPoint < P1
        {
          result[1] = nextPoint;
        }
      }
      else // nextPoint > P1, nextPoint < P2
      {
        result[2] = nextPoint;
      }
    }
    else // nextPoint > P2
    {
      // Do nothing. nextPoint not in top 3.
    }
  }

  return vtkm::ErrorCode::Success;
}

/// \brief Returns the min point id of a cell face
/// Given information about a cell face and the global point indices for that cell, returns a
/// vtkm::Id that contains the minimum point id for that face.
template <typename CellShapeTag, typename GlobalPointIndicesVecType>
static inline VTKM_EXEC vtkm::ErrorCode CellFaceMinPointId(
  vtkm::IdComponent faceIndex,
  CellShapeTag shape,
  const GlobalPointIndicesVecType& globalPointIndicesVec,
  vtkm::Id& minFacePointId)
{
  vtkm::IdComponent numPointsInFace;
  minFacePointId = { -1 };
  VTKM_RETURN_ON_ERROR(vtkm::exec::CellFaceNumberOfPoints(faceIndex, shape, numPointsInFace));
  if (numPointsInFace < 1)
  {
    // An invalid face. We should already have gotten an error from
    // CellFaceNumberOfPoints.
    return vtkm::ErrorCode::InvalidFaceId;
  }

  detail::CellFaceTables table;
  minFacePointId = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 0)];
  vtkm::Id nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 1)];
  if (nextPoint < minFacePointId)
  {
    minFacePointId = nextPoint;
  }
  nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, 2)];
  if (nextPoint < minFacePointId)
  {
    minFacePointId = nextPoint;
  }

  // Check the rest of the points to see if they are in the lowest 3
  for (vtkm::IdComponent pointIndex = 3; pointIndex < numPointsInFace; pointIndex++)
  {
    nextPoint = globalPointIndicesVec[table.PointsInFace(shape.Id, faceIndex, pointIndex)];
    if (nextPoint < minFacePointId)
    {
      minFacePointId = nextPoint;
    }
  }

  return vtkm::ErrorCode::Success;
}
}
} // namespace vtkm::exec

#endif //vtk_m_exec_CellFace_h
