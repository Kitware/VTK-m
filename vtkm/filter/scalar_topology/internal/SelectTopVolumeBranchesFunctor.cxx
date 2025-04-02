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
#include <vtkm/cont/EnvironmentTracker.h>
#include <vtkm/filter/scalar_topology/internal/SelectTopVolumeBranchesFunctor.h>
#include <vtkm/filter/scalar_topology/worklet/branch_decomposition/hierarchical_volumetric_branch_decomposer/GetOuterEndWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ArrayTransforms.h>
#include <vtkm/filter/scalar_topology/worklet/select_top_volume_branches/BranchVolumeComparator.h>

#include <vtkm/Types.h>

#include <iomanip>

#ifdef DEBUG_PRINT
#define DEBUG_PRINT_COMBINED_HIGH_VOLUME_BRANCH
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/PrintVectors.h>
#endif

namespace vtkm
{
namespace filter
{
namespace scalar_topology
{
namespace internal
{

void SelectTopVolumeBranchesFunctor::operator()(
  SelectTopVolumeBranchesBlock* b,
  const vtkmdiy::ReduceProxy& rp,     // communication proxy
  const vtkmdiy::RegularSwapPartners& // partners of the current block (unused)
) const
{
  // Get our rank and DIY id
  const vtkm::Id rank = vtkm::cont::EnvironmentTracker::GetCommunicator().rank();
  const auto selfid = rp.gid();

  // Aliases to reduce verbosity
  using IdArrayType = vtkm::worklet::contourtree_augmented::IdArrayType;

  vtkm::cont::Invoker invoke;

  std::vector<int> incoming;
  rp.incoming(incoming);
  for (const int ingid : incoming)
  {
    // NOTE/IMPORTANT: In each round we should have only one swap partner (despite for-loop here).
    // If that assumption does not hold, it will break things.
    // NOTE/IMPORTANT: This assumption only holds if the number of blocks is a power of two.
    // Otherwise, we may need to process more than one incoming block
    if (ingid != selfid)
    {
      // copy incoming to the block
#ifdef DEBUG_PRINT_COMBINED_HIGH_VOLUME_BRANCH
      int incomingGlobalBlockId;
      rp.dequeue(ingid, incomingGlobalBlockId);
      VTKM_LOG_S(vtkm::cont::LogLevel::Info,
                 "Combining local block " << b->GlobalBlockId << " with incoming block "
                                          << incomingGlobalBlockId);
#endif

      // dequeue the data from other blocks.
      // nIncomingBranches
      // array of incoming branch global regular ID
      // array of incoming branch volume
      // array of branch epsilon direction
      // array of branch upper end global regular ID
      // array of branch lower end global regular ID
      // array of branch saddle end value
      vtkm::Id nIncoming;
      rp.dequeue(ingid, nIncoming);

      IdArrayType incomingTopVolBranchGRId;
      rp.dequeue(ingid, incomingTopVolBranchGRId);

      IdArrayType incomingTopVolBranchVolume;
      rp.dequeue(ingid, incomingTopVolBranchVolume);

      IdArrayType incomingTopVolBranchSaddleEpsilon;
      rp.dequeue(ingid, incomingTopVolBranchSaddleEpsilon);

      IdArrayType incomingTopVolBranchUpperEnd;
      rp.dequeue(ingid, incomingTopVolBranchUpperEnd);

      IdArrayType incomingTopVolBranchLowerEnd;
      rp.dequeue(ingid, incomingTopVolBranchLowerEnd);

      std::stringstream dataSizeStream;
      // Log the amount of exchanged data
      dataSizeStream << "    " << std::setw(38) << std::left << "Incoming top volume branch size"
                     << ": " << nIncoming << std::endl;

      VTKM_LOG_S(this->TimingsLogLevel,
                 std::endl
                   << "    ---------------- Select Top Volume Branches Step ---------------------"
                   << std::endl
                   << "    Rank    : " << rank << std::endl
                   << "    DIY Id  : " << selfid << std::endl
                   << "    Inc Id  : " << ingid << std::endl
                   << dataSizeStream.str());

      vtkm::Id nSelf = b->TopVolumeData.TopVolumeBranchRootGRId.GetNumberOfValues();
#ifdef DEBUG_PRINT_COMBINED_HIGH_VOLUME_BRANCH
      VTKM_LOG_S(vtkm::cont::LogLevel::Info, "nIncoming = " << nIncoming << ", nSelf = " << nSelf);
      {
        std::stringstream rs;
        vtkm::worklet::contourtree_augmented::PrintHeader(nIncoming, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "incomingTopBranchId", incomingTopVolBranchGRId, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "incomingTopBranchVol", incomingTopVolBranchVolume, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintValues<ValueType>(
          "incomingSaddleVal", incomingTopVolBranchSaddleIsoValue, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "incomingUpperEnd", incomingTopVolBranchUpperEnd, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "incomingLowerEnd", incomingTopVolBranchLowerEnd, -1, rs);

        vtkm::worklet::contourtree_augmented::PrintHeader(nSelf, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "selfTopBranchId", b->TopVolumeData.TopVolumeBranchRootGRId, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "selfTopBranchVol", b->TopVolumeData.TopVolumeBranchVolume, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintValues<ValueType>(
          "selfTopSaddleVal", inArray, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "selfTopBranchUpperEnd", b->TopVolumeData.TopVolumeBranchUpperEndGRId, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "selfTopBranchLowerEnd", b->TopVolumeData.TopVolumeBranchLowerEndGRId, -1, rs);
        VTKM_LOG_S(vtkm::cont::LogLevel::Info, rs.str());
      }
#endif
      // merge incoming branches with self branches
      IdArrayType mergedTopVolBranchGRId;
      IdArrayType mergedTopVolBranchVolume;
      IdArrayType mergedTopVolBranchSaddleEpsilon;
      IdArrayType mergedTopVolBranchUpperEnd;
      IdArrayType mergedTopVolBranchLowerEnd;
      mergedTopVolBranchGRId.Allocate(nIncoming + nSelf);
      mergedTopVolBranchVolume.Allocate(nIncoming + nSelf);
      mergedTopVolBranchSaddleEpsilon.Allocate(nIncoming + nSelf);
      mergedTopVolBranchUpperEnd.Allocate(nIncoming + nSelf);
      mergedTopVolBranchLowerEnd.Allocate(nIncoming + nSelf);

      vtkm::cont::Algorithm::CopySubRange(
        incomingTopVolBranchGRId, 0, nIncoming, mergedTopVolBranchGRId, 0);
      vtkm::cont::Algorithm::CopySubRange(
        incomingTopVolBranchVolume, 0, nIncoming, mergedTopVolBranchVolume, 0);
      vtkm::cont::Algorithm::CopySubRange(
        incomingTopVolBranchSaddleEpsilon, 0, nIncoming, mergedTopVolBranchSaddleEpsilon, 0);
      vtkm::cont::Algorithm::CopySubRange(
        incomingTopVolBranchUpperEnd, 0, nIncoming, mergedTopVolBranchUpperEnd, 0);
      vtkm::cont::Algorithm::CopySubRange(
        incomingTopVolBranchLowerEnd, 0, nIncoming, mergedTopVolBranchLowerEnd, 0);

      vtkm::cont::Algorithm::CopySubRange(
        b->TopVolumeData.TopVolumeBranchRootGRId, 0, nSelf, mergedTopVolBranchGRId, nIncoming);
      vtkm::cont::Algorithm::CopySubRange(
        b->TopVolumeData.TopVolumeBranchVolume, 0, nSelf, mergedTopVolBranchVolume, nIncoming);
      vtkm::cont::Algorithm::CopySubRange(b->TopVolumeData.TopVolumeBranchSaddleEpsilon,
                                          0,
                                          nSelf,
                                          mergedTopVolBranchSaddleEpsilon,
                                          nIncoming);
      vtkm::cont::Algorithm::CopySubRange(b->TopVolumeData.TopVolumeBranchUpperEndGRId,
                                          0,
                                          nSelf,
                                          mergedTopVolBranchUpperEnd,
                                          nIncoming);
      vtkm::cont::Algorithm::CopySubRange(b->TopVolumeData.TopVolumeBranchLowerEndGRId,
                                          0,
                                          nSelf,
                                          mergedTopVolBranchLowerEnd,
                                          nIncoming);

      // Sort all branches (incoming + self) based on volume
      // sorting key: (volume, branch global regular ID)
      // the highest volume comes first, the lowest branch GR ID comes first
      vtkm::cont::ArrayHandleIndex mergedBranchId(nIncoming + nSelf);
      IdArrayType sortedBranchId;
      vtkm::cont::Algorithm::Copy(mergedBranchId, sortedBranchId);
      vtkm::worklet::scalar_topology::select_top_volume_branches::BranchVolumeComparator
        branchVolumeComparator(mergedTopVolBranchGRId, mergedTopVolBranchVolume);
      vtkm::cont::Algorithm::Sort(sortedBranchId, branchVolumeComparator);

      // permute the branch information based on sorting
      IdArrayType permutedTopVolBranchGRId;
      vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id, IdArrayType>(
        mergedTopVolBranchGRId, sortedBranchId, permutedTopVolBranchGRId);
      IdArrayType permutedTopVolBranchVolume;
      vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id, IdArrayType>(
        mergedTopVolBranchVolume, sortedBranchId, permutedTopVolBranchVolume);
      IdArrayType permutedTopVolBranchSaddleEpsilon;
      vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id, IdArrayType>(
        mergedTopVolBranchSaddleEpsilon, sortedBranchId, permutedTopVolBranchSaddleEpsilon);
      IdArrayType permutedTopVolBranchUpperEnd;
      vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id, IdArrayType>(
        mergedTopVolBranchUpperEnd, sortedBranchId, permutedTopVolBranchUpperEnd);
      IdArrayType permutedTopVolBranchLowerEnd;
      vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id, IdArrayType>(
        mergedTopVolBranchLowerEnd, sortedBranchId, permutedTopVolBranchLowerEnd);

#ifdef DEBUG_PRINT_COMBINED_HIGH_VOLUME_BRANCH
      {
        std::stringstream rs;
        vtkm::worklet::contourtree_augmented::PrintHeader(nIncoming + nSelf, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "permutedTopBranchId", permutedTopVolBranchGRId, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "permutedTopBranchVol", permutedTopVolBranchVolume, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "permutedTopBranchUpperEnd", permutedTopVolBranchUpperEnd, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "permutedTopBranchLowerEnd", permutedTopVolBranchLowerEnd, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintValues<ValueType>(
          "permutedTopSaddleVal", permutedTopVolBranchSaddleIsoValue, -1, rs);
      }
#endif

      // there may be duplicate branches. We remove duplicate branches based on global regular IDs
      // We can reuse the filter from removing duplicate branches in the process of collecting branches
      IdArrayType oneIfUniqueBranch;
      oneIfUniqueBranch.Allocate(nIncoming + nSelf);
      vtkm::worklet::scalar_topology::hierarchical_volumetric_branch_decomposer::
        OneIfBranchEndWorklet oneIfUniqueWorklet;
      invoke(oneIfUniqueWorklet, mergedBranchId, permutedTopVolBranchGRId, oneIfUniqueBranch);

      // Remove duplicate
      IdArrayType mergedUniqueBranchGRId;
      IdArrayType mergedUniqueBranchVolume;
      IdArrayType mergedUniqueBranchSaddleEpsilon;
      IdArrayType mergedUniqueBranchUpperEnd;
      IdArrayType mergedUniqueBranchLowerEnd;

      vtkm::cont::Algorithm::CopyIf(
        permutedTopVolBranchGRId, oneIfUniqueBranch, mergedUniqueBranchGRId);
      vtkm::cont::Algorithm::CopyIf(
        permutedTopVolBranchVolume, oneIfUniqueBranch, mergedUniqueBranchVolume);
      vtkm::cont::Algorithm::CopyIf(
        permutedTopVolBranchSaddleEpsilon, oneIfUniqueBranch, mergedUniqueBranchSaddleEpsilon);
      vtkm::cont::Algorithm::CopyIf(
        permutedTopVolBranchUpperEnd, oneIfUniqueBranch, mergedUniqueBranchUpperEnd);
      vtkm::cont::Algorithm::CopyIf(
        permutedTopVolBranchLowerEnd, oneIfUniqueBranch, mergedUniqueBranchLowerEnd);

      vtkm::Id nMergedUnique = mergedUniqueBranchGRId.GetNumberOfValues();

      // We move all processing to the TopVolumeBranchSaddleIsoValue here
      // to reduce the size of the [&] function
      // if the code inside the [&] function is too long, compilation for ubuntu1604+gcc5 may break
      vtkm::cont::UnknownArrayHandle tempIncomingTopVolBranchSaddleIsoValue;
      rp.dequeue(ingid, tempIncomingTopVolBranchSaddleIsoValue);

      auto resolveMerging = [&](auto& inArray) {
        using InArrayHandleType = std::decay_t<decltype(inArray)>;
        using ValueType = typename InArrayHandleType::ValueType;

        InArrayHandleType incomingTopVolBranchSaddleIsoValue =
          tempIncomingTopVolBranchSaddleIsoValue.AsArrayHandle<InArrayHandleType>();
        InArrayHandleType mergedTopVolBranchSaddleIsoValue;
        mergedTopVolBranchSaddleIsoValue.Allocate(nIncoming + nSelf);

        vtkm::cont::Algorithm::CopySubRange<ValueType, ValueType>(
          incomingTopVolBranchSaddleIsoValue, 0, nIncoming, mergedTopVolBranchSaddleIsoValue, 0);

        vtkm::cont::Algorithm::CopySubRange<ValueType, ValueType>(
          inArray, 0, nSelf, mergedTopVolBranchSaddleIsoValue, nIncoming);

        InArrayHandleType permutedTopVolBranchSaddleIsoValue;
        vtkm::worklet::contourtree_augmented::PermuteArrayWithRawIndex<InArrayHandleType>(
          mergedTopVolBranchSaddleIsoValue, sortedBranchId, permutedTopVolBranchSaddleIsoValue);

        InArrayHandleType mergedUniqueBranchSaddleIsoValue;
        vtkm::cont::Algorithm::CopyIf(
          permutedTopVolBranchSaddleIsoValue, oneIfUniqueBranch, mergedUniqueBranchSaddleIsoValue);

        if (nMergedUnique > this->NumSavedBranches)
        {
          inArray.Allocate(this->NumSavedBranches);
          vtkm::cont::Algorithm::CopySubRange(
            mergedUniqueBranchSaddleIsoValue, 0, this->NumSavedBranches, inArray);
        }
        else
        {
          inArray.Allocate(nMergedUnique);
          vtkm::cont::Algorithm::Copy(mergedUniqueBranchSaddleIsoValue, inArray);
        }
      };
      b->TopVolumeData.TopVolumeBranchSaddleIsoValue
        .CastAndCallForTypes<vtkm::TypeListScalarAll, vtkm::cont::StorageListBasic>(resolveMerging);

#ifdef DEBUG_PRINT_COMBINED_HIGH_VOLUME_BRANCH
      {
        std::stringstream rs;
        vtkm::worklet::contourtree_augmented::PrintHeader(nMergedUnique, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "mergedUniqueBranchId", mergedUniqueBranchGRId, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "mergedUniqueBranchVol", mergedUniqueBranchVolume, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "mergedUniqueBranchUpperEnd", mergedUniqueBranchUpperEnd, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "mergedUniqueBranchLowerEnd", mergedUniqueBranchLowerEnd, -1, rs);
        vtkm::worklet::contourtree_augmented::PrintValues<ValueType>(
          "mergedUniqueSaddleVal", mergedUniqueBranchSaddleIsoValue, -1, rs);
      }
#endif

      // After removing duplicate, if there are more branches than we need
      // We only save the top NumSavedBranches branches
      if (nMergedUnique > this->NumSavedBranches)
      {
        vtkm::cont::Algorithm::CopySubRange(mergedUniqueBranchGRId,
                                            0,
                                            this->NumSavedBranches,
                                            b->TopVolumeData.TopVolumeBranchRootGRId);
        vtkm::cont::Algorithm::CopySubRange(mergedUniqueBranchVolume,
                                            0,
                                            this->NumSavedBranches,
                                            b->TopVolumeData.TopVolumeBranchVolume);
        vtkm::cont::Algorithm::CopySubRange(mergedUniqueBranchSaddleEpsilon,
                                            0,
                                            this->NumSavedBranches,
                                            b->TopVolumeData.TopVolumeBranchSaddleEpsilon);
        vtkm::cont::Algorithm::CopySubRange(mergedUniqueBranchUpperEnd,
                                            0,
                                            this->NumSavedBranches,
                                            b->TopVolumeData.TopVolumeBranchUpperEndGRId);
        vtkm::cont::Algorithm::CopySubRange(mergedUniqueBranchLowerEnd,
                                            0,
                                            this->NumSavedBranches,
                                            b->TopVolumeData.TopVolumeBranchLowerEndGRId);
      }
      else
      {
        vtkm::cont::Algorithm::Copy(mergedUniqueBranchGRId,
                                    b->TopVolumeData.TopVolumeBranchRootGRId);
        vtkm::cont::Algorithm::Copy(mergedUniqueBranchVolume,
                                    b->TopVolumeData.TopVolumeBranchVolume);
        vtkm::cont::Algorithm::Copy(mergedUniqueBranchSaddleEpsilon,
                                    b->TopVolumeData.TopVolumeBranchSaddleEpsilon);
        vtkm::cont::Algorithm::Copy(mergedUniqueBranchUpperEnd,
                                    b->TopVolumeData.TopVolumeBranchUpperEndGRId);
        vtkm::cont::Algorithm::Copy(mergedUniqueBranchLowerEnd,
                                    b->TopVolumeData.TopVolumeBranchLowerEndGRId);
      }
    }
  }

  for (int cc = 0; cc < rp.out_link().size(); ++cc)
  {

    auto target = rp.out_link().target(cc);
    if (target.gid != selfid)
    {
#ifdef DEBUG_PRINT_COMBINED_HIGH_VOLUME_BRANCH
      rp.enqueue(target, b->GlobalBlockId);
      VTKM_LOG_S(vtkm::cont::LogLevel::Info,
                 "Block " << b->GlobalBlockId << " enqueue to Block " << target.gid);
#endif

      vtkm::Id nBranches = b->TopVolumeData.TopVolumeBranchRootGRId.GetNumberOfValues();

      rp.enqueue(target, nBranches);
      rp.enqueue(target, b->TopVolumeData.TopVolumeBranchRootGRId);
      rp.enqueue(target, b->TopVolumeData.TopVolumeBranchVolume);
      rp.enqueue(target, b->TopVolumeData.TopVolumeBranchSaddleEpsilon);
      rp.enqueue(target, b->TopVolumeData.TopVolumeBranchUpperEndGRId);
      rp.enqueue(target, b->TopVolumeData.TopVolumeBranchLowerEndGRId);
      rp.enqueue(target, b->TopVolumeData.TopVolumeBranchSaddleIsoValue);
    }
  }
}

} // namespace internal
} // namespace scalar_topology
} // namespace filter
} // namespace vtkm
