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
//  The PPP2 algorithm and software were jointly developed by
//  Hamish Carr (University of Leeds), Gunther H. Weber (LBNL), and
//  Oliver Ruebel (LBNL)
//==============================================================================

#ifndef vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_fill_empty_iteration_worklet_h
#define vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_fill_empty_iteration_worklet_h

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>
#include <vtkm/worklet/WorkletMapField.h>

namespace vtkm
{
namespace worklet
{
namespace contourtree_distributed
{
namespace hierarchical_augmenter
{

// Worklet for a rare case where an iteration has no supernode
// need to update the FirstSupernodePerIteration array to avoid crash
class FillEmptyIterationWorklet : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(
    WholeArrayInOut augmentedTreeFirstSupernodePerIteration // input/output
  );
  using ExecutionSignature = void(InputIndex, _1);
  using InputDomain = _1;

  VTKM_EXEC_CONT
  FillEmptyIterationWorklet() {}

  /// operator() of the worklet
  template <typename InOutFieldPortalType>
  VTKM_EXEC void operator()(const vtkm::Id& inputIndex,
                            InOutFieldPortalType& firstSupernodePerIteration) const
  {
    if (inputIndex == 0 || inputIndex == firstSupernodePerIteration.GetNumberOfValues() - 1)
      return;
    if (firstSupernodePerIteration.Get(inputIndex) == 0)
    {
      vtkm::Id nextSupernode = firstSupernodePerIteration.Get(inputIndex + 1);
      firstSupernodePerIteration.Set(inputIndex, nextSupernode);
    }
  } // operator()()
};  // FillEmptyIterationWorklet

} // namespace hierarchical_augmenter
} // namespace contourtree_distributed
} // namespace worklet
} // namespace vtkm

#endif
