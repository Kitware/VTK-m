//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_rendering_GlyphType_h
#define vtk_m_rendering_GlyphType_h

namespace vtkm
{
namespace rendering
{

/// @brief Glyph shapes supported by glyphing mappers.
enum struct GlyphType
{
  Arrow,
  Axes,
  Cube,
  Quad,
  Sphere,
};

}
}

#endif
