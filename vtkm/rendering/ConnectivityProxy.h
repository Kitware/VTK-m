//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_rendering_ConnectivityProxy_h
#define vtk_m_rendering_ConnectivityProxy_h

#include <memory>
#include <vtkm/cont/DataSet.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/Mapper.h>
#include <vtkm/rendering/View.h>
#include <vtkm/rendering/raytracing/Camera.h>
#include <vtkm/rendering/raytracing/PartialComposite.h>
#include <vtkm/rendering/raytracing/Ray.h>

namespace vtkm
{
namespace rendering
{

using PartialVector64 = std::vector<vtkm::rendering::raytracing::PartialComposite<vtkm::Float64>>;
using PartialVector32 = std::vector<vtkm::rendering::raytracing::PartialComposite<vtkm::Float32>>;

class VTKM_RENDERING_EXPORT ConnectivityProxy
{
public:
  ConnectivityProxy(const vtkm::cont::DataSet& dataset, const std::string& fieldName);

  ConnectivityProxy(const vtkm::cont::DataSet& dataSet,
                    const std::string& fieldName,
                    const std::string& coordinateName);

  ConnectivityProxy(const vtkm::cont::UnknownCellSet& cellset,
                    const vtkm::cont::CoordinateSystem& coords,
                    const vtkm::cont::Field& scalarField);

  ConnectivityProxy(const ConnectivityProxy&);
  ConnectivityProxy& operator=(const ConnectivityProxy&);

  ConnectivityProxy(ConnectivityProxy&&) noexcept;
  ConnectivityProxy& operator=(ConnectivityProxy&&) noexcept;

  ~ConnectivityProxy();

  enum struct RenderMode
  {
    Volume,
    Energy,
  };

  void SetRenderMode(RenderMode mode);
  void SetSampleDistance(const vtkm::Float32&);
  void SetScalarField(const std::string& fieldName);
  void SetEmissionField(const std::string& fieldName);
  void SetScalarRange(const vtkm::Range& range);
  void SetColorMap(vtkm::cont::ArrayHandle<vtkm::Vec4f_32>& colormap);
  void SetCompositeBackground(bool on);
  void SetDebugPrints(bool on);
  void SetUnitScalar(vtkm::Float32 unitScalar);
  void SetEpsilon(vtkm::Float64 epsilon); // epsilon for bumping lost rays

  vtkm::Bounds GetSpatialBounds();
  vtkm::Range GetScalarFieldRange();
  vtkm::Range GetScalarRange();

  void Trace(const vtkm::rendering::Camera& camera, vtkm::rendering::CanvasRayTracer* canvas);
  void Trace(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays);
  void Trace(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays);

  PartialVector64 PartialTrace(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays);
  PartialVector32 PartialTrace(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays);

protected:
  struct InternalsType;
  std::unique_ptr<InternalsType> Internals;
};
}
} //namespace vtkm::rendering
#endif //vtk_m_rendering_ConnectivityProxy_h
