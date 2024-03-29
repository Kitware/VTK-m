//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_rendering_MapperRayTracer_h
#define vtk_m_rendering_MapperRayTracer_h

#include <vtkm/cont/ColorTable.h>
#include <vtkm/rendering/Camera.h>
#include <vtkm/rendering/Mapper.h>

#include <memory>

namespace vtkm
{
namespace rendering
{

/// @brief Mapper to render surfaces using ray tracing.
///
/// Provides a "standard" data mapper that uses ray tracing to render the surfaces
/// of `DataSet` objects.
class VTKM_RENDERING_EXPORT MapperRayTracer : public Mapper
{
public:
  MapperRayTracer();

  ~MapperRayTracer();

  void SetCanvas(vtkm::rendering::Canvas* canvas) override;
  virtual vtkm::rendering::Canvas* GetCanvas() const override;

  void SetCompositeBackground(bool on);
  vtkm::rendering::Mapper* NewCopy() const override;
  void SetShadingOn(bool on);

private:
  struct InternalsType;
  std::shared_ptr<InternalsType> Internals;
  struct CompareIndices;

  void RenderCellsImpl(const vtkm::cont::UnknownCellSet& cellset,
                       const vtkm::cont::CoordinateSystem& coords,
                       const vtkm::cont::Field& scalarField,
                       const vtkm::cont::ColorTable& colorTable,
                       const vtkm::rendering::Camera& camera,
                       const vtkm::Range& scalarRange,
                       const vtkm::cont::Field& ghostField) override;
};
}
} //namespace vtkm::rendering

#endif //vtk_m_rendering_MapperRayTracer_h
