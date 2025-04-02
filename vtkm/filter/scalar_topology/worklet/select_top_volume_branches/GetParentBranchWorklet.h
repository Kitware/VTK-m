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

#ifndef vtk_m_filter_scalar_topology_worklet_select_top_volume_branches_GetParentBranchWorklet_h
#define vtk_m_filter_scalar_topology_worklet_select_top_volume_branches_GetParentBranchWorklet_h

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>
#include <vtkm/worklet/WorkletMapField.h>

namespace vtkm
{
namespace worklet
{
namespace scalar_topology
{
namespace select_top_volume_branches
{

constexpr vtkm::IdComponent MAX_CONNECTIVITY_3D = static_cast<vtkm::IdComponent>(14);
using IdArrayType = vtkm::worklet::contourtree_augmented::IdArrayType;

/// <summary>
/// worklet to compute the parent branch of branches
/// </summary>
class GetParentBranchWorklet : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(
    FieldIn branchSaddle,     // (input) branch saddle supernode id
    FieldIn saddleBranchRoot, // (input) the branch root of the superarc starting from the saddle
    FieldIn saddleGRId,       // (input) branch saddle supernode global regular id
    WholeArrayIn superarcs,   // (array input) all superarc targets in ascending order
    WholeArrayIn branchRoots, // (array input) all branchRoots of superarcs
    WholeArrayIn branchRootByBranch, // (array input) branch roots of branches in ascending order
    WholeArrayIn upperEndGRIds,      // (array input) upper local end of branches
    WholeArrayIn lowerEndGRIds,      // (array input) lower local end of branches
    FieldOut parentBranch            // (output) the information index of the parent branch
  );
  using ExecutionSignature = _9(_1, _2, _3, _4, _5, _6, _7, _8);
  using InputDomain = _1;

  /// Constructor
  VTKM_EXEC_CONT
  GetParentBranchWorklet() {}

  template <typename IdArrayPortalType>
  VTKM_EXEC vtkm::Id GetSuperarcEndPoint(const vtkm::Id& branchSaddle,
                                         const IdArrayPortalType& sortedSuperarcs,
                                         const bool isStart) const
  {
    VTKM_ASSERT(vtkm::worklet::contourtree_augmented::NoSuchElement(sortedSuperarcs.Get(0)));
    using vtkm::worklet::contourtree_augmented::MaskedIndex;
    vtkm::Id nSuperarcs = sortedSuperarcs.GetNumberOfValues();
    vtkm::Id endpoint = vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT;
    vtkm::Id head = 1;
    vtkm::Id tail = nSuperarcs - 1;
    while (head <= tail)
    {
      vtkm::Id mid = (head + tail) >> 1;
      vtkm::Id midSuperarc = MaskedIndex(sortedSuperarcs.Get(mid));
      if (midSuperarc > branchSaddle)
        tail = mid - 1;
      else if (midSuperarc < branchSaddle)
        head = mid + 1;
      else if (isStart &&
               (mid == 1 || (mid > 1 && MaskedIndex(sortedSuperarcs.Get(mid - 1)) < branchSaddle)))
      {
        endpoint = mid;
        break;
      }
      else if (!isStart &&
               (mid == nSuperarcs - 1 ||
                (mid < nSuperarcs - 1 && MaskedIndex(sortedSuperarcs.Get(mid + 1)) > branchSaddle)))
      {
        endpoint = mid;
        break;
      }
      else if (isStart)
        tail = mid - 1;
      else
        head = mid + 1;
    }
    VTKM_ASSERT(endpoint >= 1);
    return endpoint;
  }

  template <typename IdArrayPortalType>
  VTKM_EXEC vtkm::Id GetBranchRootIdx(const vtkm::Id& branchRoot,
                                      const IdArrayPortalType& branchRootByBranch) const
  {
    vtkm::Id nBranchRoot = branchRootByBranch.GetNumberOfValues();
    vtkm::Id head = 0;
    vtkm::Id tail = nBranchRoot - 1;
    while (head <= tail)
    {
      vtkm::Id mid = (head + tail) >> 1;
      vtkm::Id midBranchRoot = branchRootByBranch.Get(mid);
      if (midBranchRoot == branchRoot)
      {
        return mid;
      }
      else if (midBranchRoot > branchRoot)
        tail = mid - 1;
      else
        head = mid + 1;
    }
    // Update 10/04/2024:
    // we use this binary search to filter the removed superarcs/branches in presimplification,
    // so we do not report error, but use NO_SUCH_ELEMENT to indicate the nonexistence of the branch
    return vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT;
  }

  template <typename IdArrayPortalType>
  VTKM_EXEC vtkm::Id operator()(const vtkm::Id& branchSaddle,
                                const vtkm::Id& saddleBranchRoot,
                                const vtkm::Id& saddleGRId,
                                const IdArrayPortalType& sortedSuperarcs,
                                const IdArrayPortalType& permutedBranchRoots,
                                const IdArrayPortalType& branchRootByBranch,
                                const IdArrayPortalType& upperEndGRIds,
                                const IdArrayPortalType& lowerEndGRIds) const
  {
    // there are at most MAX_CONNECTIVITY_3D superarcs connected to the branchSaddle
    vtkm::Id candidateBranchRoot[MAX_CONNECTIVITY_3D];
    vtkm::Id nCandidate = 1;
    candidateBranchRoot[0] = saddleBranchRoot;

    const vtkm::Id superarcStartIdx = GetSuperarcEndPoint(branchSaddle, sortedSuperarcs, true);
    const vtkm::Id superarcEndIdx = GetSuperarcEndPoint(branchSaddle, sortedSuperarcs, false);
    VTKM_ASSERT(superarcEndIdx >= superarcStartIdx);
    // avoid integer type overflow
    VTKM_ASSERT(superarcEndIdx - superarcStartIdx + 2 > 0);
    VTKM_ASSERT(superarcEndIdx - superarcStartIdx + 2 <= MAX_CONNECTIVITY_3D);
    for (vtkm::Id superarc = superarcStartIdx; superarc <= superarcEndIdx; superarc++)
    {
      candidateBranchRoot[nCandidate++] = permutedBranchRoots.Get(superarc);
    }

    for (vtkm::Id branchRoot = 0; branchRoot < nCandidate; branchRoot++)
    {
      // Update 10/04/2024:
      // The superarc starting from the saddle may not be valid,
      // because it can be a virtual superarc or a presimplified superarc.
      // We use the indicator (branchIdx == NO_SUCH_ELEMENT) to handle it.
      const vtkm::Id branchIdx =
        GetBranchRootIdx(candidateBranchRoot[branchRoot], branchRootByBranch);
      if (vtkm::worklet::contourtree_augmented::NoSuchElement(branchIdx))
        continue;
      if (upperEndGRIds.Get(branchIdx) != saddleGRId && lowerEndGRIds.Get(branchIdx) != saddleGRId)
        return branchIdx;
    }

    // Unfortunately, it seems possible that the parent branch cannot be found
    // in which case NO_SUCH_ELEMENT is returned
    // VTKM_ASSERT(false && "Cannot find the parent branch!");
    return vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT;
  }
}; // GetParentBranchWorklet

} // namespace select_top_volume_branches
} // namespace scalar_topology
} // namespace worklet
} // namespace vtkm

#endif
