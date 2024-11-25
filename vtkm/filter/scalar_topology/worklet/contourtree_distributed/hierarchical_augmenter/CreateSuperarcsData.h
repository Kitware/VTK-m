//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
// Copyright (c) 2018, The Regents of the University of California, through
// Lawrence Berkeley National Laboratory (subject to receipt of any required approvals
// from the U.S. Dept. of Energy).  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence Berkeley National
//     Laboratory, U.S. Dept. of Energy nor the names of its contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//
//=============================================================================
//
//  This code is an extension of the algorithm presented in the paper:
//  Parallel Peak Pruning for Scalable SMP Contour Tree Computation.
//  Hamish Carr, Gunther Weber, Christopher Sewell, and James Ahrens.
//  Proceedings of the IEEE Symposium on Large Data Analysis and Visualization
//  (LDAV), October 2016, Baltimore, Maryland.
//
//  The PPP2 algorithm and software were jointly developed by
//  Hamish Carr (University of Leeds), Gunther H. Weber (LBNL), and
//  Oliver Ruebel (LBNL)
//==============================================================================

// This header contains an Execution Object used to pass a arrays to the
// CreateSuperarcsWorklet to overcome the limitation of 20 input parameters for a worklet

#ifndef vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_create_superarcs_data_h
#define vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_create_superarcs_data_h

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>

#include <vtkm/cont/ExecutionObjectBase.h>

namespace vtkm
{
namespace worklet
{
namespace contourtree_distributed
{
namespace hierarchical_augmenter
{


class CreateSuperarcsData
{
public:
  // Sort indicies types
  using IndicesPortalType = vtkm::worklet::contourtree_augmented::IdArrayType::ReadPortalType;

  VTKM_EXEC_CONT
  CreateSuperarcsData() {}

  VTKM_CONT
  CreateSuperarcsData(
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeHyperparents,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeWhichRound,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeWhichIteration,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSupernodes,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSuperarcs,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSuperparents,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSuper2Hypernode,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeHypernodes,
    const vtkm::worklet::contourtree_augmented::IdArrayType& superparentSet,
    const vtkm::worklet::contourtree_augmented::IdArrayType& newSupernodeIds,
    vtkm::cont::DeviceAdapterId device,
    vtkm::cont::Token& token)
  {
    this->BaseTreeHyperparents = baseTreeHyperparents.PrepareForInput(device, token);
    this->BaseTreeWhichRound = baseTreeWhichRound.PrepareForInput(device, token);
    this->BaseTreeWhichIteration = baseTreeWhichIteration.PrepareForInput(device, token);
    this->BaseTreeSupernodes = baseTreeSupernodes.PrepareForInput(device, token);
    this->BaseTreeSuperarcs = baseTreeSuperarcs.PrepareForInput(device, token);
    this->BaseTreeSuperparents = baseTreeSuperparents.PrepareForInput(device, token);
    this->BaseTreeSuper2Hypernode = baseTreeSuper2Hypernode.PrepareForInput(device, token);
    this->BaseTreeHypernodes = baseTreeHypernodes.PrepareForInput(device, token);
    this->SuperparentSet = superparentSet.PrepareForInput(device, token);
    this->NewSupernodeIds = newSupernodeIds.PrepareForInput(device, token);
  }

public:
  IndicesPortalType BaseTreeHyperparents;
  IndicesPortalType BaseTreeWhichRound;
  IndicesPortalType BaseTreeWhichIteration;
  IndicesPortalType BaseTreeSupernodes;
  IndicesPortalType BaseTreeSuperarcs;
  IndicesPortalType BaseTreeSuperparents;
  IndicesPortalType BaseTreeSuper2Hypernode;
  IndicesPortalType BaseTreeHypernodes;
  IndicesPortalType SuperparentSet;
  IndicesPortalType NewSupernodeIds;
};


class CreateSuperarcsDataExec : public vtkm::cont::ExecutionObjectBase
{
public:
  VTKM_EXEC_CONT
  CreateSuperarcsDataExec(
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeHyperparents,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeWhichRound,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeWhichIteration,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSupernodes,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSuperarcs,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSuperparents,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeSuper2Hypernode,
    const vtkm::worklet::contourtree_augmented::IdArrayType& baseTreeHypernodes,
    const vtkm::worklet::contourtree_augmented::IdArrayType& superparentSet,
    const vtkm::worklet::contourtree_augmented::IdArrayType& newSupernodeIds)
    : BaseTreeHyperparents(baseTreeHyperparents)
    , BaseTreeWhichRound(baseTreeWhichRound)
    , BaseTreeWhichIteration(baseTreeWhichIteration)
    , BaseTreeSupernodes(baseTreeSupernodes)
    , BaseTreeSuperarcs(baseTreeSuperarcs)
    , BaseTreeSuperparents(baseTreeSuperparents)
    , BaseTreeSuper2Hypernode(baseTreeSuper2Hypernode)
    , BaseTreeHypernodes(baseTreeHypernodes)
    , SuperparentSet(superparentSet)
    , NewSupernodeIds(newSupernodeIds)
  {
  }

  VTKM_CONT
  CreateSuperarcsData PrepareForExecution(vtkm::cont::DeviceAdapterId device,
                                          vtkm::cont::Token& token) const
  {
    return CreateSuperarcsData(BaseTreeHyperparents,
                               BaseTreeWhichRound,
                               BaseTreeWhichIteration,
                               BaseTreeSupernodes,
                               BaseTreeSuperarcs,
                               BaseTreeSuperparents,
                               BaseTreeSuper2Hypernode,
                               BaseTreeHypernodes,
                               SuperparentSet,
                               NewSupernodeIds,
                               device,
                               token);
  }

private:
  // Whole array data used from the BaseTree in CreateSuperarcsWorklet
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeHyperparents;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeWhichRound;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeWhichIteration;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeSupernodes;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeSuperarcs;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeSuperparents;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeSuper2Hypernode;
  const vtkm::worklet::contourtree_augmented::IdArrayType& BaseTreeHypernodes;
  const vtkm::worklet::contourtree_augmented::IdArrayType& SuperparentSet;
  const vtkm::worklet::contourtree_augmented::IdArrayType& NewSupernodeIds;
};



} // namespace hierarchical_augmenter
} // namespace contourtree_distributed
} // namespace worklet
} // namespace vtkm

#endif
