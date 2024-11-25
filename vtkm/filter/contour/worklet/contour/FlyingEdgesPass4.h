
//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================


#ifndef vtk_m_worklet_contour_flyingedges_pass4_h
#define vtk_m_worklet_contour_flyingedges_pass4_h

#include <vtkm/filter/contour/worklet/contour/FlyingEdgesPass4Common.h>
#include <vtkm/filter/contour/worklet/contour/FlyingEdgesPass4X.h>
#include <vtkm/filter/contour/worklet/contour/FlyingEdgesPass4XWithNormals.h>
#include <vtkm/filter/contour/worklet/contour/FlyingEdgesPass4Y.h>

namespace vtkm
{
namespace worklet
{
namespace flying_edges
{

struct launchComputePass4
{
  vtkm::Id3 PointDims;

  vtkm::Id CellWriteOffset;
  vtkm::Id PointWriteOffset;

  launchComputePass4(const vtkm::Id3& pdims,
                     vtkm::Id multiContourCellOffset,
                     vtkm::Id multiContourPointOffset)
    : PointDims(pdims)
    , CellWriteOffset(multiContourCellOffset)
    , PointWriteOffset(multiContourPointOffset)
  {
  }

  template <typename DeviceAdapterTag,
            typename IVType,
            typename T,
            typename CoordsType,
            typename StorageTagField,
            typename MeshSums,
            typename PointType,
            typename NormalType>
  VTKM_CONT bool LaunchXAxis(DeviceAdapterTag device,
                             vtkm::Id vtkmNotUsed(newPointSize),
                             IVType isoval,
                             CoordsType coordinateSystem,
                             const vtkm::cont::ArrayHandle<T, StorageTagField>& inputField,
                             vtkm::cont::ArrayHandle<vtkm::UInt8> edgeCases,
                             vtkm::cont::CellSetStructured<2>& metaDataMesh2D,
                             const MeshSums& metaDataSums,
                             const vtkm::cont::ArrayHandle<vtkm::Id>& metaDataMin,
                             const vtkm::cont::ArrayHandle<vtkm::Id>& metaDataMax,
                             const vtkm::cont::ArrayHandle<vtkm::Int32>& metaDataNumTris,
                             vtkm::worklet::contour::CommonState& sharedState,
                             vtkm::cont::ArrayHandle<vtkm::Id>& triangle_topology,
                             PointType& points,
                             NormalType& normals) const
  {
    vtkm::cont::Invoker invoke(device);
    if (sharedState.GenerateNormals)
    {
      ComputePass4XWithNormals<IVType> worklet4(
        isoval, this->PointDims, this->CellWriteOffset, this->PointWriteOffset);
      invoke(worklet4,
             metaDataMesh2D,
             metaDataSums,
             metaDataMin,
             metaDataMax,
             metaDataNumTris,
             edgeCases,
             coordinateSystem,
             inputField,
             triangle_topology,
             sharedState.InterpolationEdgeIds,
             sharedState.InterpolationWeights,
             sharedState.CellIdMap,
             points,
             normals);
    }
    else
    {
      ComputePass4X<IVType> worklet4(
        isoval, this->PointDims, this->CellWriteOffset, this->PointWriteOffset);
      invoke(worklet4,
             metaDataMesh2D,
             metaDataSums,
             metaDataMin,
             metaDataMax,
             metaDataNumTris,
             edgeCases,
             coordinateSystem,
             inputField,
             triangle_topology,
             sharedState.InterpolationEdgeIds,
             sharedState.InterpolationWeights,
             sharedState.CellIdMap,
             points);
    }

    return true;
  }

  template <typename DeviceAdapterTag,
            typename IVType,
            typename T,
            typename CoordsType,
            typename StorageTagField,
            typename MeshSums,
            typename PointType,
            typename NormalType>
  VTKM_CONT bool LaunchYAxis(DeviceAdapterTag device,
                             vtkm::Id newPointSize,
                             IVType isoval,
                             CoordsType coordinateSystem,
                             const vtkm::cont::ArrayHandle<T, StorageTagField>& inputField,
                             vtkm::cont::ArrayHandle<vtkm::UInt8> edgeCases,
                             vtkm::cont::CellSetStructured<2>& metaDataMesh2D,
                             const MeshSums& metaDataSums,
                             const vtkm::cont::ArrayHandle<vtkm::Id>& metaDataMin,
                             const vtkm::cont::ArrayHandle<vtkm::Id>& metaDataMax,
                             const vtkm::cont::ArrayHandle<vtkm::Int32>& metaDataNumTris,
                             vtkm::worklet::contour::CommonState& sharedState,
                             vtkm::cont::ArrayHandle<vtkm::Id>& triangle_topology,
                             PointType& points,
                             NormalType& normals) const
  {
    vtkm::cont::Invoker invoke(device);

    ComputePass4Y<IVType> worklet4(
      isoval, this->PointDims, this->CellWriteOffset, this->PointWriteOffset);
    invoke(worklet4,
           metaDataMesh2D,
           metaDataSums,
           metaDataMin,
           metaDataMax,
           metaDataNumTris,
           edgeCases,
           inputField,
           triangle_topology,
           sharedState.InterpolationEdgeIds,
           sharedState.InterpolationWeights,
           sharedState.CellIdMap);

    //This needs to be done on array handle view ( start = this->PointWriteOffset, len = newPointSize)
    ComputePass5Y<IVType> worklet5(
      this->PointDims, this->PointWriteOffset, sharedState.GenerateNormals);

    invoke(worklet5,
           vtkm::cont::make_ArrayHandleView(
             sharedState.InterpolationEdgeIds, this->PointWriteOffset, newPointSize),
           vtkm::cont::make_ArrayHandleView(
             sharedState.InterpolationWeights, this->PointWriteOffset, newPointSize),
           vtkm::cont::make_ArrayHandleView(points, this->PointWriteOffset, newPointSize),
           inputField,
           coordinateSystem,
           normals);

    return true;
  }

  template <typename DeviceAdapterTag, typename... Args>
  VTKM_CONT bool Launch(SumXAxis, DeviceAdapterTag device, Args&&... args) const
  {
    return this->LaunchXAxis(device, std::forward<Args>(args)...);
  }

  template <typename DeviceAdapterTag, typename... Args>
  VTKM_CONT bool Launch(SumYAxis, DeviceAdapterTag device, Args&&... args) const
  {
    return this->LaunchYAxis(device, std::forward<Args>(args)...);
  }

  template <typename DeviceAdapterTag, typename... Args>
  VTKM_CONT bool operator()(DeviceAdapterTag device, Args&&... args) const
  {
    return this->Launch(
      typename select_AxisToSum<DeviceAdapterTag>::type{}, device, std::forward<Args>(args)...);
  }
};
}
}
}
#endif
