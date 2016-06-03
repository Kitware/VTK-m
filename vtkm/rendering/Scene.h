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
#ifndef vtk_m_rendering_Scene_h
#define vtk_m_rendering_Scene_h

#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/Camera.h>
#include <vector>

namespace vtkm {
namespace rendering {

class Scene
{
public:
  std::vector<vtkm::rendering::Actor> Actors;

  Scene() {}

  template<typename MapperType, typename SurfaceType>
  VTKM_CONT_EXPORT
  void Render(MapperType &mapper,
              SurfaceType &surface,
              vtkm::rendering::Camera &camera)
  {
    vtkm::Bounds bounds;

    mapper.StartScene();
    for (std::size_t i = 0; i < this->Actors.size(); i++)
    {
      this->Actors[i].Render(mapper, surface, camera);

      // accumulate all Actors' spatial bounds into the scene spatial bounds
      bounds.Include(this->Actors[i].SpatialBounds);
    }
    mapper.EndScene();

    this->SpatialBounds = bounds;
  }

  const vtkm::Bounds &GetSpatialBounds()
  {
    return this->SpatialBounds;
  }

protected:
  vtkm::Bounds SpatialBounds;
};

}} //namespace vtkm::rendering

#endif //vtk_m_rendering_Scene_h
