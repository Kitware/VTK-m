//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2015 Sandia Corporation.
//  Copyright 2015 UT-Battelle, LLC.
//  Copyright 2015 Los Alamos National Security.
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================
#ifndef vtk_m_rendering_raytracing_Cell_Intersector_h
#define vtk_m_rendering_raytracing_Cell_Intersector_h

#include <vtkm/rendering/raytracing/CellTables.h>
#include <vtkm/rendering/raytracing/TriangleIntersector.h>

namespace vtkm
{
namespace rendering
{
namespace raytracing
{
//
// Any supported element. If the cell shape is not
// supported it does nothing, e.g. a 2D cellkk.
//
template <typename T>
VTKM_EXEC_CONT inline void IntersectZoo(T xpoints[8],
                                        T ypoints[8],
                                        T zpoints[8],
                                        const vtkm::Vec<T, 3>& dir,
                                        const vtkm::Vec<T, 3>& origin,
                                        T distances[6],
                                        const vtkm::Int32& shapeType)
{
  // Some precalc for water tight intersections
  T sx, sy, sz;
  vtkm::Int32 kx, ky, kz;
  WaterTight<T> intersector;
  intersector.FindDir(dir, sx, sy, sz, kx, ky, kz);
  const vtkm::Int32 tableOffset = ZooLookUp[CellTypeLookUp[shapeType]][0];
  const vtkm::Int32 numTriangles = ZooLookUp[CellTypeLookUp[shapeType]][1];
  // Decompose each face into two triangles
  for (int i = 0; i < 6; ++i)
    distances[i] = -1.;
  for (int i = 0; i < numTriangles; ++i)
  {
    const vtkm::Int32 offset = tableOffset + i;
    vtkm::Vec<T, 3> a, c, b;
    a[0] = xpoints[ZooTable[offset][1]];
    a[1] = ypoints[ZooTable[offset][1]];
    a[2] = zpoints[ZooTable[offset][1]];
    b[0] = xpoints[ZooTable[offset][2]];
    b[1] = ypoints[ZooTable[offset][2]];
    b[2] = zpoints[ZooTable[offset][2]];
    c[0] = xpoints[ZooTable[offset][3]];
    c[1] = ypoints[ZooTable[offset][3]];
    c[2] = zpoints[ZooTable[offset][3]];
    const vtkm::Int32 faceId = ZooTable[offset][0];
    T distance = -1.f;

    T uNotUsed, vNotUsed;
    intersector.IntersectTriSn(a,
                               b,
                               c,
                               sx,
                               sy,
                               sz,
                               kx,
                               ky,
                               kz,
                               distance,
                               uNotUsed,
                               vNotUsed,
                               origin[0],
                               origin[1],
                               origin[2]);

    if (distance != -1.f)
    {
      if (distances[faceId] != -1.f)
        distances[faceId] = vtkm::Min(distance, distances[faceId]);
      else
        distances[faceId] = distance;
    }
  }
}
template <typename T>
VTKM_EXEC_CONT inline void IntersectHex(T xpoints[8],
                                        T ypoints[8],
                                        T zpoints[8],
                                        const vtkm::Vec<T, 3>& dir,
                                        const vtkm::Vec<T, 3>& origin,
                                        T distances[6])
{
  // Some precalc for water tight intersections
  T sx, sy, sz;
  vtkm::Int32 kx, ky, kz;
  WaterTight<T> intersector;
  intersector.FindDir(dir, sx, sy, sz, kx, ky, kz);
  // Decompose each face into two triangles
  for (int i = 0; i < 6; ++i)
  {
    vtkm::Vec<T, 3> a, c, b, d;
    a[0] = xpoints[ShapesFaceList[i][1]];
    a[1] = ypoints[ShapesFaceList[i][1]];
    a[2] = zpoints[ShapesFaceList[i][1]];
    b[0] = xpoints[ShapesFaceList[i][2]];
    b[1] = ypoints[ShapesFaceList[i][2]];
    b[2] = zpoints[ShapesFaceList[i][2]];
    c[0] = xpoints[ShapesFaceList[i][3]];
    c[1] = ypoints[ShapesFaceList[i][3]];
    c[2] = zpoints[ShapesFaceList[i][3]];
    d[0] = xpoints[ShapesFaceList[i][4]];
    d[1] = ypoints[ShapesFaceList[i][4]];
    d[2] = zpoints[ShapesFaceList[i][4]];
    T distance = -1.f;
    distances[i] = distance; //init to -1

    T uNotUsed, vNotUsed;
    intersector.IntersectTriSn(a,
                               b,
                               c,
                               sx,
                               sy,
                               sz,
                               kx,
                               ky,
                               kz,
                               distance,
                               uNotUsed,
                               vNotUsed,
                               origin[0],
                               origin[1],
                               origin[2]);

    if (distance != -1.f)
      distances[i] = distance;

    distance = -1.f;

    intersector.IntersectTriSn(a,
                               c,
                               d,
                               sx,
                               sy,
                               sz,
                               kx,
                               ky,
                               kz,
                               distance,
                               uNotUsed,
                               vNotUsed,
                               origin[0],
                               origin[1],
                               origin[2]);



    if (distance != -1.f)
    {
      if (distances[i] != -1.f)
        distances[i] = vtkm::Min(distance, distances[i]);
      else
        distances[i] = distance;
    }
  }
}
template <typename T>
VTKM_EXEC_CONT inline void IntersectTet(T xpoints[8],
                                        T ypoints[8],
                                        T zpoints[8],
                                        const vtkm::Vec<T, 3>& dir,
                                        const vtkm::Vec<T, 3>& origin,
                                        T distances[6])
{
  // Some precalc for water tight intersections
  T sx, sy, sz;
  vtkm::Int32 kx, ky, kz;
  WaterTight<T> intersector;
  intersector.FindDir(dir, sx, sy, sz, kx, ky, kz);

  const vtkm::Int32 tableOffset = FaceLookUp[CellTypeLookUp[CELL_SHAPE_TETRA]][0];
  for (vtkm::Int32 i = 0; i < 4; ++i)
  {
    vtkm::Vec<T, 3> a, c, b;
    a[0] = xpoints[ShapesFaceList[i + tableOffset][1]];
    a[1] = ypoints[ShapesFaceList[i + tableOffset][1]];
    a[2] = zpoints[ShapesFaceList[i + tableOffset][1]];
    b[0] = xpoints[ShapesFaceList[i + tableOffset][2]];
    b[1] = ypoints[ShapesFaceList[i + tableOffset][2]];
    b[2] = zpoints[ShapesFaceList[i + tableOffset][2]];
    c[0] = xpoints[ShapesFaceList[i + tableOffset][3]];
    c[1] = ypoints[ShapesFaceList[i + tableOffset][3]];
    c[2] = zpoints[ShapesFaceList[i + tableOffset][3]];
    T distance = -1.f;
    distances[i] = distance; //init to -1

    T uNotUsed, vNotUsed;

    intersector.IntersectTriSn(a,
                               b,
                               c,
                               sx,
                               sy,
                               sz,
                               kx,
                               ky,
                               kz,
                               distance,
                               uNotUsed,
                               vNotUsed,
                               origin[0],
                               origin[1],
                               origin[2]);

    if (distance != -1.f)
      distances[i] = distance;
  }
}

//
// Wedge
//
template <typename T>
VTKM_EXEC_CONT inline void IntersectWedge(T xpoints[8],
                                          T ypoints[8],
                                          T zpoints[8],
                                          const vtkm::Vec<T, 3>& dir,
                                          const vtkm::Vec<T, 3>& origin,
                                          T distances[6])
{
  // Some precalc for water tight intersections
  T sx, sy, sz;
  vtkm::Int32 kx, ky, kz;
  WaterTight<T> intersector;
  intersector.FindDir(dir, sx, sy, sz, kx, ky, kz);
  // TODO: try two sepate loops to see performance impact
  const vtkm::Int32 tableOffset = FaceLookUp[CellTypeLookUp[CELL_SHAPE_WEDGE]][0];
  // Decompose each face into two triangles
  for (int i = 0; i < 5; ++i)
  {
    vtkm::Vec<T, 3> a, c, b, d;
    a[0] = xpoints[ShapesFaceList[i + tableOffset][1]];
    a[1] = ypoints[ShapesFaceList[i + tableOffset][1]];
    a[2] = zpoints[ShapesFaceList[i + tableOffset][1]];
    b[0] = xpoints[ShapesFaceList[i + tableOffset][2]];
    b[1] = ypoints[ShapesFaceList[i + tableOffset][2]];
    b[2] = zpoints[ShapesFaceList[i + tableOffset][2]];
    c[0] = xpoints[ShapesFaceList[i + tableOffset][3]];
    c[1] = ypoints[ShapesFaceList[i + tableOffset][3]];
    c[2] = zpoints[ShapesFaceList[i + tableOffset][3]];
    d[0] = xpoints[ShapesFaceList[i + tableOffset][4]];
    d[1] = ypoints[ShapesFaceList[i + tableOffset][4]];
    d[2] = zpoints[ShapesFaceList[i + tableOffset][4]];
    T distance = -1.f;
    distances[i] = distance; //init to -1

    T uNotUsed, vNotUsed;

    intersector.IntersectTriSn(a,
                               b,
                               c,
                               sx,
                               sy,
                               sz,
                               kx,
                               ky,
                               kz,
                               distance,
                               uNotUsed,
                               vNotUsed,
                               origin[0],
                               origin[1],
                               origin[2]);

    if (distance != -1.f)
      distances[i] = distance;
    //
    //First two faces are triangles
    //
    if (i < 2)
      continue;
    distance = -1.f;

    intersector.IntersectTriSn(a,
                               c,
                               d,
                               sx,
                               sy,
                               sz,
                               kx,
                               ky,
                               kz,
                               distance,
                               uNotUsed,
                               vNotUsed,
                               origin[0],
                               origin[1],
                               origin[2]);

    if (distance != -1.f)
    {
      if (distances[i] != -1.f)
        distances[i] = vtkm::Min(distance, distances[i]);
      else
        distances[i] = distance;
    }
  }
}

//
// General Template should never be instantiated
//

template <int CellType>
class CellIntersector
{
public:
  template <typename T>
  VTKM_EXEC_CONT inline void IntersectCell(T* vtkmNotUsed(xpoints),
                                           T* vtkmNotUsed(ypoints),
                                           T* vtkmNotUsed(zpoints),
                                           const vtkm::Vec<T, 3>& vtkmNotUsed(dir),
                                           const vtkm::Vec<T, 3>& vtkmNotUsed(origin),
                                           T* vtkmNotUsed(distances),
                                           const vtkm::UInt8 vtkmNotUsed(cellShape = 12));
};

//
// Hex Specialization
//
template <>
class CellIntersector<CELL_SHAPE_HEXAHEDRON>
{
public:
  template <typename T>
  VTKM_EXEC_CONT inline void IntersectCell(T* xpoints,
                                           T* ypoints,
                                           T* zpoints,
                                           const vtkm::Vec<T, 3>& dir,
                                           const vtkm::Vec<T, 3>& origin,
                                           T* distances,
                                           const vtkm::UInt8 cellShape = 12) const
  {
    if (cellShape == 12)
    {
      IntersectZoo(xpoints, ypoints, zpoints, dir, origin, distances, cellShape);
    }
    else
    {
      printf("CellIntersector Hex Error: unsupported cell type. Doing nothing\n");
    }
  }
};
//
//
// Hex Specialization Structured
//
template <>
class CellIntersector<254>
{
public:
  template <typename T>
  VTKM_EXEC_CONT inline void IntersectCell(T* xpoints,
                                           T* ypoints,
                                           T* zpoints,
                                           const vtkm::Vec<T, 3>& dir,
                                           const vtkm::Vec<T, 3>& origin,
                                           T* distances,
                                           const vtkm::UInt8 cellShape = 12) const
  {
    if (cellShape == 12)
    {
      IntersectZoo(xpoints, ypoints, zpoints, dir, origin, distances, cellShape);
    }
    else
    {
      printf("CellIntersector Hex Error: unsupported cell type. Doing nothing\n");
    }
  }
};

//
// Tet Specialization
//
template <>
class CellIntersector<CELL_SHAPE_TETRA>
{
public:
  template <typename T>
  VTKM_EXEC_CONT inline void IntersectCell(T* xpoints,
                                           T* ypoints,
                                           T* zpoints,
                                           const vtkm::Vec<T, 3>& dir,
                                           const vtkm::Vec<T, 3>& origin,
                                           T distances[6],
                                           const vtkm::UInt8 cellShape = 12) const
  {
    if (cellShape == CELL_SHAPE_TETRA)
    {
      IntersectTet(xpoints, ypoints, zpoints, dir, origin, distances);
    }
    else
    {
      printf("CellIntersector Tet Error: unsupported cell type. Doing nothing\n");
    }
  }
};

//
// Wedge Specialization
//
template <>
class CellIntersector<CELL_SHAPE_WEDGE>
{
public:
  template <typename T>
  VTKM_EXEC_CONT inline void IntersectCell(T* xpoints,
                                           T* ypoints,
                                           T* zpoints,
                                           const vtkm::Vec<T, 3>& dir,
                                           const vtkm::Vec<T, 3>& origin,
                                           T distances[6],
                                           const vtkm::UInt8 cellShape = 12) const
  {
    if (cellShape == CELL_SHAPE_WEDGE)
    {
      IntersectWedge(xpoints, ypoints, zpoints, dir, origin, distances);
    }
    else
    {
      printf("CellIntersector Wedge Error: unsupported cell type. Doing nothing\n");
    }
  }
};
//
// Zoo elements
//
template <>
class CellIntersector<255>
{
public:
  template <typename T>
  VTKM_EXEC_CONT inline void IntersectCell(T* xpoints,
                                           T* ypoints,
                                           T* zpoints,
                                           const vtkm::Vec<T, 3>& dir,
                                           const vtkm::Vec<T, 3>& origin,
                                           T distances[6],
                                           const vtkm::UInt8 cellShape = 0) const
  {
    IntersectZoo(xpoints, ypoints, zpoints, dir, origin, distances, cellShape);
  }
};
}
}
} // namespace vtkm::rendering::raytracing
#endif
