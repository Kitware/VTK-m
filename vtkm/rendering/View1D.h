//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_rendering_View1D_h
#define vtk_m_rendering_View1D_h

#include <vtkm/rendering/AxisAnnotation2D.h>
#include <vtkm/rendering/ColorLegendAnnotation.h>
#include <vtkm/rendering/View.h>

namespace vtkm
{
namespace rendering
{

/// @brief A view for a 1D data set.
///
/// 1D data are rendered as an X-Y plot with the values shone on the Y axis.
class VTKM_RENDERING_EXPORT View1D : public vtkm::rendering::View
{
public:
  View1D(const vtkm::rendering::Scene& scene,
         const vtkm::rendering::Mapper& mapper,
         const vtkm::rendering::Canvas& canvas,
         const vtkm::rendering::Color& backgroundColor = vtkm::rendering::Color(0, 0, 0, 1),
         const vtkm::rendering::Color& foregroundColor = vtkm::rendering::Color(1, 1, 1, 1));

  View1D(const vtkm::rendering::Scene& scene,
         const vtkm::rendering::Mapper& mapper,
         const vtkm::rendering::Canvas& canvas,
         const vtkm::rendering::Camera& camera,
         const vtkm::rendering::Color& backgroundColor = vtkm::rendering::Color(0, 0, 0, 1),
         const vtkm::rendering::Color& foregroundColor = vtkm::rendering::Color(1, 1, 1, 1));

  void Paint() override;
  void RenderScreenAnnotations() override;
  void RenderWorldAnnotations() override;
  void RenderColorLegendAnnotations();

  void EnableLegend();
  void DisableLegend();
  void SetLegendLabelColor(vtkm::rendering::Color c) { this->Legend.SetLabelColor(c); }

  /// @brief Specify whether log scaling should be used on the X axis.
  void SetLogX(bool l)
  {
    this->GetMapper().SetLogarithmX(l);
    this->LogX = l;
  }

  /// @brief Specify whether log scaling should be used on the Y axis.
  void SetLogY(bool l)
  {
    this->GetMapper().SetLogarithmY(l);
    this->LogY = l;
  }

private:
  void UpdateCameraProperties();

  // 1D-specific annotations
  vtkm::rendering::AxisAnnotation2D HorizontalAxisAnnotation;
  vtkm::rendering::AxisAnnotation2D VerticalAxisAnnotation;
  vtkm::rendering::ColorLegendAnnotation Legend;
  bool LegendEnabled = true;
  bool LogX = false;
  bool LogY = false;
};
}
} // namespace vtkm::rendering

#endif //vtk_m_rendering_View1D_h
