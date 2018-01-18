//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2016 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2016 UT-Battelle, LLC.
//  Copyright 2016 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#include <vtkm/rendering/ColorBarAnnotation.h>
#include <vtkm/rendering/TextAnnotationScreen.h>

namespace vtkm
{
namespace rendering
{

ColorBarAnnotation::ColorBarAnnotation()
{
  vtkm::Bounds bounds(vtkm::Range(-0.88, +0.88), vtkm::Range(+0.87, +0.92), vtkm::Range(0, 0));
  Position = bounds;
  Horizontal = true;
  FieldName = "";
}

ColorBarAnnotation::~ColorBarAnnotation()
{
}

void ColorBarAnnotation::SetFieldName(const std::string& fieldName)
{
  FieldName = fieldName;
}

void ColorBarAnnotation::SetPosition(const vtkm::Bounds& position)
{
  Position = position;
  vtkm::Float64 x = Position.X.Length();
  vtkm::Float64 y = Position.Y.Length();
  if (x > y)
    Horizontal = true;
  else
    Horizontal = false;
}

void ColorBarAnnotation::SetRange(const vtkm::Range& range, vtkm::IdComponent numTicks)
{
  std::vector<vtkm::Float64> positions, proportions;
  this->Axis.SetMinorTicks(positions, proportions); // clear any minor ticks

  for (vtkm::IdComponent i = 0; i < numTicks; ++i)
  {
    vtkm::Float64 prop = static_cast<vtkm::Float64>(i) / static_cast<vtkm::Float64>(numTicks - 1);
    vtkm::Float64 pos = range.Min + prop * range.Length();
    positions.push_back(pos);
    proportions.push_back(prop);
  }
  this->Axis.SetMajorTicks(positions, proportions);
}

void ColorBarAnnotation::Render(const vtkm::rendering::Camera& camera,
                                const vtkm::rendering::WorldAnnotator& worldAnnotator,
                                vtkm::rendering::Canvas& canvas)
{

  canvas.AddColorBar(Position, this->ColorTable, Horizontal);

  this->Axis.SetColor(canvas.GetForegroundColor());
  this->Axis.SetLineWidth(1);

  if (Horizontal)
  {
    this->Axis.SetScreenPosition(Position.X.Min, Position.Y.Min, Position.X.Max, Position.Y.Min);
    this->Axis.SetLabelAlignment(TextAnnotation::HCenter, TextAnnotation::Top);
    this->Axis.SetMajorTickSize(0, .02, 1.0);
  }
  else
  {
    this->Axis.SetScreenPosition(Position.X.Min, Position.Y.Min, Position.X.Min, Position.Y.Max);
    this->Axis.SetLabelAlignment(TextAnnotation::Right, TextAnnotation::VCenter);
    this->Axis.SetMajorTickSize(.02, 0.0, 1.0);
  }

  this->Axis.SetMinorTickSize(0, 0, 0); // no minor ticks
  this->Axis.Render(camera, worldAnnotator, canvas);

  if (FieldName != "")
  {
    vtkm::Vec<vtkm::Float32, 2> labelPos;
    if (Horizontal)
    {
      labelPos[0] = vtkm::Float32(Position.X.Min);
      labelPos[1] = vtkm::Float32(Position.Y.Max);
    }
    else
    {
      labelPos[0] = vtkm::Float32(Position.X.Min - 0.07);
      labelPos[1] = vtkm::Float32(Position.Y.Max + 0.03);
    }

    vtkm::rendering::TextAnnotationScreen var(FieldName,
                                              canvas.GetForegroundColor(),
                                              .045f, // font scale
                                              labelPos,
                                              0.f); // rotation

    var.Render(camera, worldAnnotator, canvas);
  }
}
}
} // namespace vtkm::rendering
