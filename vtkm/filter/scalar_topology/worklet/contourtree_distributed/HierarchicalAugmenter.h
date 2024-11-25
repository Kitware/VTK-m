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
//=======================================================================================
//
//  Parallel Peak Pruning v. 2.0
//
//  Started June 15, 2017
//
// Copyright Hamish Carr, University of Leeds
//
// HierarchicalAugmenter.h
//
//=======================================================================================
//
// COMMENTS:
//
//  In order to compute geometric measures properly, we probably need to have all supernodes
//  inserted rather than the lazy insertion implicit in the existing computation.  After discussion,
//  we have decided to make this a post-processing step in order to keep our options open.
//
//  Fortunately, the HierarchicalContourTree structure will hold a tree augmented with lower level
//  supernodes, so we just want a factory class that takes a HCT as input and produces another one
//  as output.  Note that the output will no longer have insertions to be made, as all subtrees will
//  be rooted at a supernode in the parent level.
//
//  Since this is blockwise, we will have the main loop external (as with the HierarchicalHyperSweeper)
//  and have it invoke subroutines here
//
//  The processing will be based on a fanin with partners as usual
//  I.  Each block swaps all attachment points for the level with its partner
//  II.  Fanning-in builds sets of all attachment points to insert into each superarc except the base level
//  III.At the end of the fanin, we know the complete set of all supernodes to be inserted in all superarcs,
//    so we insert them all at once & renumber.  We should not need to do so in a fan-out
//
//  After some prototyping in Excel, the test we will need to apply is the following:
//
//  In round N, we transfer all attachment points whose round is < N+1 and whose superparent round is >= N+1
//  (NB: In excel, I started with Round 1, when I should have started with round 0 to keep the swap-partner correct)
//
//  The superparent round is the round at which the attachment point will be inserted at the end, so the
//  attachment point needs to be shared at all levels up to and including that round, hence the second branch
//  of the test.
//
//  The first branch is because the attachment points at round N are already represented in the partner
//  due to the construction of the hierarchical contour tree. Therefore transferring them is redundant and
//  complicates processing, so we omit them.  For higher levels, they will need to be inserted.
//
//  This test is independent of things such as sort order, so we can keep our arrays in unsorted order.
//  We may wish to revisit this later to enforce a canonical order for validation / verification
//  but as long as we are consistent, we should in fact have a canonical ordering on each block.
//
//=======================================================================================

#ifndef vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_h
#define vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_h

#include <iomanip>
#include <string>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/NotNoSuchElementPredicate.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/PrintVectors.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/PrintGraph.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/AttachmentAndSupernodeComparator.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/AttachmentIdsEqualComparator.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/AttachmentSuperparentAndIndexComparator.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/CopyBaseRegularStructureWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/CreateSuperarcsData.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/CreateSuperarcsSetFirstSupernodePerIterationWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/CreateSuperarcsUpdateFirstSupernodePerIterationWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/CreateSuperarcsWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/FindSuperparentForNecessaryNodesWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/HierarchicalAugmenterInOutData.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/IsAscendingDecorator.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/IsAttachementPointNeededPredicate.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/IsAttachementPointPredicate.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/ResizeArraysBuildNewSupernodeIdsWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/SetFirstAttachmentPointInRoundWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/SetSuperparentSetDecorator.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/UpdateHyperstructureSetHyperarcsAndNodesWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_augmenter/UpdateHyperstructureSetSuperchildrenWorklet.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/hierarchical_contour_tree/PermuteComparator.h>

#ifdef DEBUG_PRINT
#define DEBUG_PRINT_HIERARCHICAL_AUGMENTER
#define DEBUG_PRINT_HIERARCHICAL_CONTOUR_TREE
#endif

namespace vtkm
{
namespace worklet
{
namespace contourtree_distributed
{

/// Facture class for augmenting the hierarchical contour tree to enable computations of measures, e.g., volumne
template <typename FieldType>
class HierarchicalAugmenter
{ // class HierarchicalAugmenter
public:
  /// base mesh variable needs to determine whether a vertex is inside or outside of the block
  vtkm::Id3 MeshBlockOrigin;
  vtkm::Id3 MeshBlockSize;
  vtkm::Id3 MeshGlobalSize;

  /// the tree that it hypersweeps over
  vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>* BaseTree;
  /// the tree that it is building
  vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>* AugmentedTree;

  /// the ID of the base block (used for debug output)
  vtkm::Id BlockId;

  /// arrays storing the details for the attachment points & old supernodes the Id in the global data set
  vtkm::worklet::contourtree_augmented::IdArrayType GlobalRegularIds;

  /// the data value
  vtkm::cont::ArrayHandle<FieldType> DataValues;

  /// the supernode index
  /// when we swap attachment points, we will set this to NO_SUCH_ELEMENT because the supernodes
  /// added are on a different block, so their original supernodeId becomes invalid
  vtkm::worklet::contourtree_augmented::IdArrayType SupernodeIds;

  /// the superarc will *ALWAYS* be -1 for a true attachment point, so we don't store it
  /// instead, the superparent stores the Id for the arc it inserts into
  /// WARNING: in order for sorting to work, we will need to carry forward the ascending / descending flag
  /// This flag is normally stored on the superarc, but will be stored in this class on the superparent
  vtkm::worklet::contourtree_augmented::IdArrayType Superparents;

  /// we want to track the round on which the superparent is transferred (we could look it up, but it's
  /// more convenient to have it here). Also, we don't need the iteration.
  vtkm::worklet::contourtree_augmented::IdArrayType SuperparentRounds;

  /// we also want to track the round on which the attachment point was originally transferred
  vtkm::worklet::contourtree_augmented::IdArrayType WhichRounds;

  /// if we're not careful, we'll have read-write conflicts when swapping with the partner
  /// there are other solutions, but the simpler solution is to have a transfer buffer for
  /// the set we want to send - which means another set of parallel arrays
  /// Output arrays used in DIY exchange
  vtkm::worklet::contourtree_distributed::hierarchical_augmenter::HierarchicalAugmenterInOutData<
    FieldType>
    OutData;
  /// Output arrays used in DIY exchange
  vtkm::worklet::contourtree_distributed::hierarchical_augmenter::HierarchicalAugmenterInOutData<
    FieldType>
    InData;

  /// these are essentially temporary local variables, but are placed here to make the DebugPrint()
  /// more comprehensive. They will be allocated where used
  /// this one makes a list of attachment Ids and is used in several different places, so resize it when done
  vtkm::worklet::contourtree_augmented::IdArrayType AttachmentIds;
  /// tracks segments of attachment points by round
  vtkm::worklet::contourtree_augmented::IdArrayType FirstAttachmentPointInRound;
  /// maps from old supernode Id to new supernode Id
  vtkm::worklet::contourtree_augmented::IdArrayType NewSupernodeIds;
  /// tracks which supernodes are kept in a given round
  vtkm::worklet::contourtree_augmented::IdArrayType KeptSupernodes;
  /// sorting array & arrays for data details
  vtkm::worklet::contourtree_augmented::IdArrayType SupernodeSorter;
  vtkm::worklet::contourtree_augmented::IdArrayType GlobalRegularIdSet;
  vtkm::cont::ArrayHandle<FieldType> DataValueSet;
  vtkm::worklet::contourtree_augmented::IdArrayType SuperparentSet;
  vtkm::worklet::contourtree_augmented::IdArrayType SupernodeIdSet;
  /// data for transferring regular nodes
  vtkm::worklet::contourtree_augmented::IdArrayType RegularSuperparents;
  vtkm::worklet::contourtree_augmented::IdArrayType RegularNodesNeeded;

  /// empty constructor
  HierarchicalAugmenter() {}

  /// initializer (called explicitly after constructor)
  void Initialize(
    vtkm::Id blockId,
    vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>* inBaseTree,
    vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>* inAugmentedTree,
    vtkm::Id3 meshBlockOrigin,
    vtkm::Id3 meshBockSize,
    vtkm::Id3 meshGlobalSize,
    vtkm::worklet::contourtree_augmented::IdArrayType* volumeArray = NULL,
    vtkm::Id presimplifyThreshold = 0);

  /// routine to prepare the set of attachment points to transfer
  void PrepareOutAttachmentPoints(vtkm::Id round);

  /// routine to retrieve partner's current list of attachment points
  void RetrieveInAttachmentPoints();

  /// routine to release memory used for swap arrays
  void ReleaseSwapArrays();

  /// routine to reconstruct a hierarchical tree using the augmenting supernodes
  void BuildAugmentedTree();

  // subroutines for BuildAugmentedTree
  /// initial preparation
  void PrepareAugmentedTree();
  /// transfer of hyperstructure but not superchildren count
  void CopyHyperstructure();
  /// transfer level of superstructure with insertions
  void CopySuperstructure();
  /// reset the super Ids in the hyperstructure to the new values
  void UpdateHyperstructure(vtkm::Id roundNumber);
  /// copy the remaining base level regular nodes
  void CopyBaseRegularStructure();

  // subroutines for CopySuperstructure
  /// gets a list of all the old supernodes to transfer at this level (ie except attachment points
  void RetrieveOldSupernodes(vtkm::Id roundNumber);
  /// resizes the arrays for the level
  void ResizeArrays(vtkm::Id roundNumber);
  /// adds a round full of superarcs (and regular nodes) to the tree
  void CreateSuperarcs(vtkm::Id roundNumber);

  /// debug routine
  std::string DebugPrint(std::string message, const char* fileName, long lineNum);
  void DebugSave(std::string filename);

private:
  /// Used internally to Invoke worklets
  vtkm::cont::Invoker Invoke;

}; // class HierarchicalAugmenter



// initalizating function
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::Initialize(
  vtkm::Id blockId,
  vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>* baseTree,
  vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>* augmentedTree,
  vtkm::Id3 meshBlockOrigin,
  vtkm::Id3 meshBockSize,
  vtkm::Id3 meshGlobalSize,
  vtkm::worklet::contourtree_augmented::IdArrayType* volumeArray,
  vtkm::Id presimplifyThreshold)
{ // Initialize()
  // copy the parameters for use
  this->BlockId = blockId;
  this->BaseTree = baseTree;
  this->AugmentedTree = augmentedTree;
  this->MeshBlockOrigin = meshBlockOrigin;
  this->MeshBlockSize = meshBockSize;
  this->MeshGlobalSize = meshGlobalSize;

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  std::stringstream headStr;
  headStr << "=======================" << std::endl;
  headStr << "Initializing Block " << blockId << std::endl;
  headStr << "=======================" << std::endl;
  VTKM_LOG_S(vtkm::cont::LogLevel::Info, headStr.str());
#endif

  // now construct a list of all attachment points on the block
  // except those under the presimplify threshold. The presimplification is
  // handled in the IsAttachementPointPredicate

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  bool presimplify = ((volumeArray != NULL) && (presimplifyThreshold > 0));
  std::stringstream presimpStr;
  presimpStr << "Presimplification Threshold: " << presimplifyThreshold << std::endl;
  presimpStr << "Volume Array:                " << ((volumeArray != NULL) ? "T" : "F") << std::endl;
  presimpStr << "Threshold:                   " << ((presimplifyThreshold > 0) ? "T" : "F")
             << std::endl;
  presimpStr << "Presimplify: " << (presimplify ? "T" : "F") << std::endl;

  if (presimplify)
  { // presimplifying
    vtkm::worklet::contourtree_augmented::PrintHeader((*volumeArray).GetNumberOfValues(),
                                                      presimpStr);
    vtkm::worklet::contourtree_augmented::PrintIndices("Volumes: ", *volumeArray, -1, presimpStr);
    vtkm::worklet::contourtree_augmented::PrintHeader(baseTree->Superparents.GetNumberOfValues(),
                                                      presimpStr);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Global Regular", baseTree->RegularNodeGlobalIds, -1, presimpStr);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Superparents", baseTree->Superparents, -1, presimpStr);
  } // presimplifying
  VTKM_LOG_S(vtkm::cont::LogLevel::Info, presimpStr.str());
#endif

  // to do this, we construct an index array with all supernode ID's that satisfy:
  // 1. superparent == NO_SUCH_ELEMENT (i.e. root of interior tree)
  // 2. round < nRounds (except the top level, where 1. indicates the tree root)
  // initialize AttachementIds
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::IsAttachementPointPredicate
      isAttachementPointPredicate(this->BaseTree->Superarcs,
                                  this->BaseTree->WhichRound,
                                  this->BaseTree->NumRounds,
                                  volumeArray,
                                  presimplifyThreshold);
    auto tempSupernodeIndex =
      vtkm::cont::ArrayHandleIndex(this->BaseTree->Supernodes.GetNumberOfValues());

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
    // This debug routine is a serial implementation of the IsAttachmentPointPredicate
    // with necessary debug output in the intermediate steps.
    {
      std::stringstream isAttachmentStream;
      vtkm::cont::Algorithm::Copy(tempSupernodeIndex, this->AttachmentIds);
      isAttachmentStream << "Block: " << blockId << std::endl;
      vtkm::worklet::contourtree_augmented::PrintHeader(this->AttachmentIds.GetNumberOfValues(),
                                                        isAttachmentStream);
      vtkm::worklet::contourtree_augmented::PrintIndices(
        "Attachment ID", this->AttachmentIds, -1, isAttachmentStream);

      auto supernodes = this->BaseTree->Supernodes.ReadPortal();
      auto superarcs = this->BaseTree->Superarcs.ReadPortal();
      auto whichRound = this->BaseTree->WhichRound.ReadPortal();
      auto volumeArr =
        presimplify ? volumeArray->ReadPortal() : this->BaseTree->WhichRound.ReadPortal();
      auto regNodeGlobalIds = this->BaseTree->RegularNodeGlobalIds.ReadPortal();
      auto attachmentIds = this->AttachmentIds.WritePortal();
      for (vtkm::Id supernode = 0; supernode < this->AttachmentIds.GetNumberOfValues(); supernode++)
      { // per supernode
        isAttachmentStream << "Processing supernode " << supernode << std::endl;
        isAttachmentStream << "Regular ID           " << supernodes.Get(supernode) << std::endl;
        isAttachmentStream << "Global Regular ID    "
                           << regNodeGlobalIds.Get(supernodes.Get(supernode)) << std::endl;

        // an attachment point is defined by having no superarc (NO_SUCH_ELEMENT) and not being in the final round (where this indicates the global root)
        if (vtkm::worklet::contourtree_augmented::NoSuchElement(superarcs.Get(supernode)) &&
            (whichRound.Get(supernode) < baseTree->NumRounds))
        // passes the predicate - nothing further needed
        { // passes the predicate
          isAttachmentStream << "Attachment Point: it passed the first test" << std::endl;
          if (presimplify)
          {
            isAttachmentStream << "Volume:      " << volumeArr.Get(supernode) << std::endl;
            isAttachmentStream << "Threshold:   " << presimplifyThreshold << std::endl;
          }

          // suppress if it's volume is at or below the threshold
          if (presimplify && volumeArr.Get(supernode) <= presimplifyThreshold)
          { // below threshold
            attachmentIds.Set(supernode, vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
            isAttachmentStream << "Failed Second Test" << std::endl;
          } // below threshold
          else
            isAttachmentStream << "Volume Greater than Threshold: it passed the second test"
                               << std::endl;
          isAttachmentStream << "Block: " << blockId << std::endl;
          vtkm::worklet::contourtree_augmented::PrintHeader(attachmentIds.GetNumberOfValues(),
                                                            isAttachmentStream);
          vtkm::worklet::contourtree_augmented::PrintIndices(
            "Attachment ID", this->AttachmentIds, -1, isAttachmentStream);
        } // passes the predicate
        else
        { // fails
          // reset the value
          attachmentIds.Set(supernode, vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
        } // fails
      }   // per supernode

      isAttachmentStream << "Block: " << blockId << std::endl;
      vtkm::worklet::contourtree_augmented::PrintHeader(this->AttachmentIds.GetNumberOfValues(),
                                                        isAttachmentStream);
      vtkm::worklet::contourtree_augmented::PrintIndices(
        "Attachment ID", this->AttachmentIds, -1, isAttachmentStream);
      VTKM_LOG_S(vtkm::cont::LogLevel::Info, isAttachmentStream.str());
    }
#endif

    vtkm::cont::Algorithm::CopyIf(
      // first a list of all of the supernodes
      tempSupernodeIndex,
      // then our stencil
      tempSupernodeIndex,
      // And the CopyIf compress the supernodes array to eliminate the non-attachement points and
      // save to this->AttachmentIds
      this->AttachmentIds,
      // then our predicate identifies all attachment points
      // i.e., an attachment point is defined by having no superarc (NO_SUCH_ELEMENT) and not
      // being in the final round (where this indicates the global root) defined by the condition
      // if (noSuchElement(baseTree->superarcs[supernode]) && (baseTree->whichRound[supernode] < baseTree->nRounds))
      isAttachementPointPredicate);

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
    {
      std::stringstream debugStream;
      debugStream << "Block: " << blockId << std::endl;
      vtkm::worklet::contourtree_augmented::PrintHeader(this->AttachmentIds.GetNumberOfValues(),
                                                        debugStream);
      vtkm::worklet::contourtree_augmented::PrintIndices(
        "Attachment ID", this->AttachmentIds, -1, debugStream);
      VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
    }
#endif
  }

  // we now resize the working arrays
  this->GlobalRegularIds.Allocate(this->AttachmentIds.GetNumberOfValues());
  this->DataValues.Allocate(this->AttachmentIds.GetNumberOfValues());
  this->SupernodeIds.Allocate(this->AttachmentIds.GetNumberOfValues());
  this->Superparents.Allocate(this->AttachmentIds.GetNumberOfValues());
  this->SuperparentRounds.Allocate(this->AttachmentIds.GetNumberOfValues());
  this->WhichRounds.Allocate(this->AttachmentIds.GetNumberOfValues());

  // and do an indexed copy (permutation) to copy in the attachment point information
  {
    auto hierarchicalRegularIds =
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->BaseTree->Supernodes);
    auto superparents =
      vtkm::cont::make_ArrayHandlePermutation(hierarchicalRegularIds, this->BaseTree->Superparents);
    // globalRegularIDs[attachmentPoint]   = baseTree->regularNodeGlobalIDs[hierarchicalRegularID];
    vtkm::cont::Algorithm::Copy(vtkm::cont::make_ArrayHandlePermutation(
                                  hierarchicalRegularIds, BaseTree->RegularNodeGlobalIds),
                                this->GlobalRegularIds);
    //dataValues[attachmentPoint]     = baseTree->dataValues      [hierarchicalRegularID];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(hierarchicalRegularIds, BaseTree->DataValues),
      this->DataValues);
    //supernodeIDs[attachmentPoint]    = supernodeID;
    vtkm::cont::Algorithm::Copy(this->AttachmentIds, // these are our supernodeIds
                                this->SupernodeIds);
    //superparentRounds[attachmentPoint]  = baseTree->whichRound      [superparent];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(superparents, this->BaseTree->WhichRound),
      this->SuperparentRounds);
    //whichRounds[attachmentPoint]    = baseTree->whichRound[supernodeID];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->BaseTree->WhichRound),
      this->WhichRounds);

    // get the ascending flag from the superparent's superarc and transfer to the superparent
    // Array decorator to add the IS_ASCENDING flag to our superparent, i.e.,
    // if (isAscending(baseTree->superarcs[superparent])){ superparent |= IS_ASCENDING; }
    // NOTE: When using the superparents ArrayHandlePermutation in the ArrayHandleDecorator the compiler
    //       has issues discovering the StorageType when calling Copy(isAscendingSuperparentArr, superparents)
    //       by copying superparents to an actual array in tempArrSuperparents we can avoid this issue,
    //       at the cost of an extra copy.
    vtkm::worklet::contourtree_augmented::IdArrayType tempArrSuperparents;
    vtkm::cont::Algorithm::Copy(superparents, tempArrSuperparents);
    auto isAscendingSuperparentArr = vtkm::cont::make_ArrayHandleDecorator(
      tempArrSuperparents.GetNumberOfValues(), //superparents.GetNumberOfValues(),
      vtkm::worklet::contourtree_distributed::hierarchical_augmenter::IsAscendingDecorator{},
      tempArrSuperparents, //superparents,
      this->BaseTree->Superarcs);
    vtkm::cont::Algorithm::Copy(isAscendingSuperparentArr, this->Superparents);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  {
    std::stringstream debugStream;
    debugStream << "Block: " << blockId << std::endl;
    vtkm::worklet::contourtree_augmented::PrintHeader(this->AttachmentIds.GetNumberOfValues(),
                                                      debugStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Attachment ID", this->AttachmentIds, -1, debugStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Global Regular ID", this->GlobalRegularIds, -1, debugStream);
    vtkm::worklet::contourtree_augmented::PrintValues<FieldType>(
      "Data Value", this->DataValues, -1, debugStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Supernode ID", this->SupernodeIds, -1, debugStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Superparent ID ", this->Superparents, -1, debugStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Superparent Round", this->SuperparentRounds, -1, debugStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "Which Round", this->WhichRounds, -1, debugStream);
    debugStream << std::endl;
    VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
  }
#endif

  // clean up memory
  this->AttachmentIds.ReleaseResources();
} // Initialize()



// routine to prepare the set of attachment points to transfer
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::PrepareOutAttachmentPoints(vtkm::Id round)
{ // PrepareOutAttachmentPoints()
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      IsAttachementPointNeededPredicate isAttachementPointNeededPredicate(
        this->SuperparentRounds, this->WhichRounds, round);
    auto tempAttachmentPointsIndex =
      vtkm::cont::ArrayHandleIndex(this->GlobalRegularIds.GetNumberOfValues());
    vtkm::cont::Algorithm::CopyIf(
      // 1. fancy array of all of the attachment points of which parts are copied to this->AttachmentIds
      tempAttachmentPointsIndex, // input
      // 2. stencil used with the predicate to decide which AttachementIds to keep
      tempAttachmentPointsIndex, // stencil
      // 3. CopyIf compress the supernodes array to eliminate the non-attachement
      //    points and save to this->AttachmentIds
      this->AttachmentIds,
      // 4. the unary predicate uses the stencil to identify all attachment points needed
      isAttachementPointNeededPredicate);
  }

  //  4.  resize the out array. We don't need to allocate here because in step 5. the Copy
  //      algorithm will do the initalization for us

  //  5.  copy the points we want
  {
    // outGlobalRegularIDs[outAttachmentPoint]  =  globalRegularIDs[attachmentPoint];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->GlobalRegularIds),
      this->OutData.GlobalRegularIds);
    // outDataValues[outAttachmentPoint]  =  dataValues[attachmentPoint];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->DataValues),
      this->OutData.DataValues);
    // outSupernodeIDs[outAttachmentPoint]  =  supernodeIDs[attachmentPoint];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->SupernodeIds),
      this->OutData.SupernodeIds);
    // outSuperparents[outAttachmentPoint]  =  superparents[attachmentPoint];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->Superparents),
      this->OutData.Superparents);
    // outSuperparentRounds[outAttachmentPoint]  =  superparentRounds[attachmentPoint];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->SuperparentRounds),
      this->OutData.SuperparentRounds);
    // outWhichRounds[outAttachmentPoint]  =  whichRounds[attachmentPoint];
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::make_ArrayHandlePermutation(this->AttachmentIds, this->WhichRounds),
      this->OutData.WhichRounds);
  }

  // clean up memory
  this->AttachmentIds.ReleaseResources();
} // PrepareOutAttachmentPoints()


// routine to add partner's current list of attachment points
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::RetrieveInAttachmentPoints()
{ // RetrieveInAttachmentPoints()
  // what we want to do here is to copy all of the partner's attachments for the round into our own buffer
  // this code will be replaced in the MPI with a suitable transmit / receive
  // store the current size
  vtkm::Id numAttachmentsCurrently = this->GlobalRegularIds.GetNumberOfValues();
  vtkm::Id numIncomingAttachments = InData.GlobalRegularIds.GetNumberOfValues();
  vtkm::Id numTotalAttachments = numAttachmentsCurrently + numIncomingAttachments;

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  {
    std::stringstream debugStream;
    debugStream << "nAttachmentsCurrently: " << numAttachmentsCurrently << std::endl;
    debugStream << "nIncomingAttachments:  " << numIncomingAttachments << std::endl;
    debugStream << "nTotalAttachments:     " << numTotalAttachments << std::endl;
    VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
  }
#endif

  // I.  resize the existing arrays
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->GlobalRegularIds, numTotalAttachments, static_cast<vtkm::Id>(0));
  vtkm::worklet::contourtree_augmented::ResizeVector<FieldType>(
    this->DataValues, numTotalAttachments, static_cast<FieldType>(0));
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->SupernodeIds, numTotalAttachments, static_cast<vtkm::Id>(0));
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->Superparents, numTotalAttachments, static_cast<vtkm::Id>(0));
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->SuperparentRounds, numTotalAttachments, static_cast<vtkm::Id>(0));
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->WhichRounds, numTotalAttachments, static_cast<vtkm::Id>(0));

  /*this->GlobalRegularIds.Allocate(numTotalAttachments);
  this->DataValues.Allocate(numTotalAttachments);
  this->SupernodeIds.Allocate(numTotalAttachments);
  this->Superparents.Allocate(numTotalAttachments);
  this->SuperparentRounds.Allocate(numTotalAttachments);
  this->WhichRounds.Allocate(numTotalAttachments);*/

  // II. copy the additional points into them
  {
    // The following sequence of copy operations implements the following for from the orginal code
    // for (vtkm::Id inAttachmentPoint = 0; inAttachmentPoint < inGlobalRegularIDs.size(); inAttachmentPoint++)
    // globalRegularIDs[attachmentPoint]  =  inGlobalRegularIDs[outAttachmentPoint];
    auto tempGlobalRegularIdsView = vtkm::cont::make_ArrayHandleView(
      this->GlobalRegularIds, numAttachmentsCurrently, numIncomingAttachments);
    vtkm::cont::Algorithm::Copy(InData.GlobalRegularIds, tempGlobalRegularIdsView);
    // dataValues[attachmentPoint]  =  inDataValues[outAttachmentPoint];
    auto tempDataValuesView = vtkm::cont::make_ArrayHandleView(
      this->DataValues, numAttachmentsCurrently, numIncomingAttachments);
    vtkm::cont::Algorithm::Copy(InData.DataValues, tempDataValuesView);
    // supernodeIDs[attachmentPoint]  =  NO_SUCH_ELEMENT;
    auto tempNoSuchElementArr = vtkm::cont::make_ArrayHandleConstant(
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT, numIncomingAttachments);
    auto tempSupernodeIdsView = vtkm::cont::make_ArrayHandleView(
      this->SupernodeIds, numAttachmentsCurrently, numIncomingAttachments);
    vtkm::cont::Algorithm::Copy(tempNoSuchElementArr, tempSupernodeIdsView);
    // superparents[attachmentPoint]  =  inSuperparents[outAttachmentPoint];
    auto tempSuperparentsView = vtkm::cont::make_ArrayHandleView(
      this->Superparents, numAttachmentsCurrently, numIncomingAttachments);
    vtkm::cont::Algorithm::Copy(InData.Superparents, tempSuperparentsView);
    // superparentRounds[attachmentPoint]  =  inSuperparentRounds[outAttachmentPoint];
    auto tempSuperparentRoundsView = vtkm::cont::make_ArrayHandleView(
      this->SuperparentRounds, numAttachmentsCurrently, numIncomingAttachments);
    vtkm::cont::Algorithm::Copy(InData.SuperparentRounds, tempSuperparentRoundsView);
    // whichRounds[attachmentPoint]  =  inWhichRounds[outAttachmentPoint];
    auto tempWhichRoundsView = vtkm::cont::make_ArrayHandleView(
      this->WhichRounds, numAttachmentsCurrently, numIncomingAttachments);
    vtkm::cont::Algorithm::Copy(InData.WhichRounds, tempWhichRoundsView);
  }
} // RetrieveInAttachmentPoints()


// routine to release memory used for out arrays
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::ReleaseSwapArrays()
{ // ReleaseSwapArrays()
  // Rather than explicitly delete the arrays, we are going to "forget" them and
  // just release our reference count on them. If no one else is using them, the
  // memory will actually be deleted. But if an array is being used, it will
  // continue to be managed until it is not.
  this->OutData = decltype(this->OutData){};
  this->InData = decltype(this->InData){};
} // ReleaseSwapArrays()


// routine to reconstruct a hierarchical tree using the augmenting supernodes.
// Allowing pre-simplification needs the superstructure and hyperstructure to be done one layer
// at a time. This means lifting the code from CopySuperstructure up to here CopyHyperstructure()
// can actually be left the way it is - copying the entire hyperstructure,  as the augmentation
// process doesn't change it.  It might be sensible to change it anyway, just to make the code better organised
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::BuildAugmentedTree()
{ // BuildAugmentedTree()
  // 1.  Prepare the data structures for filling in, copying in basic information & organising the attachment points
  this->PrepareAugmentedTree();
#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Block ") + std::to_string(this->BlockId) +
                          std::string(" Step 1: Augmented Tree Prepared"),
                        __FILE__,
                        __LINE__));
#endif

  // 2.  Copy the hyperstructure, using the old super IDs for now
  this->CopyHyperstructure();
#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Block ") + std::to_string(this->BlockId) +
                          std::string(" Step 2: Hyperstructure Copied"),
                        __FILE__,
                        __LINE__));
#endif

  // 3.	Copy superstructure one round at a time, updating the hyperstructure as well
  // (needed to permit search for superarcs)
  //	Loop from the top down:
  for (vtkm::Id roundNumber = this->BaseTree->NumRounds; roundNumber >= 0; roundNumber--)
  { // per round
    // start by retrieving list of old supernodes from the tree (except for attachment points)
    this->RetrieveOldSupernodes(roundNumber);
    // since we know the number of attachment points, we can now allocate space for the level
    // and set up arrays for sorting the supernodes
    this->ResizeArrays(roundNumber);
    // now we create the superarcs for the round in the new tree
    this->CreateSuperarcs(roundNumber);
    // finally, we update the hyperstructure for the round in the new tree
    this->UpdateHyperstructure(roundNumber);
  } // per round
  // 4.  Copy the remaining regular structure at the bottom level, setting up the regular sort order in the process
  this->CopyBaseRegularStructure();
} // BuildAugmentedTree()


// initial preparation
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::PrepareAugmentedTree()
{ // PrepareAugmentedTree()
  //  1.  Sort attachment points on superparent round, with secondary sort on global index so duplicates appear next to each other
  //     This can (and does) happen when a vertex on the boundary is an attachment point separately for multiple blocks
  //    We add a tertiary sort on supernode ID so that on each block, it gets the correct "home" supernode ID for reconciliation
  //: note that we use a standard comparator that tie breaks with index. This separates into
  //    segments with identical superparent round, which is all we need for now
  vtkm::cont::Algorithm::Copy(
    vtkm::cont::ArrayHandleIndex(this->GlobalRegularIds.GetNumberOfValues()), this->AttachmentIds);
#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint("Attachment Points List Constructed", __FILE__, __LINE__));
#endif
  // 1a.  We now need to suppress duplicates,
  {
    // Sort the attachement Ids
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      AttachmentSuperparentAndIndexComparator attachmentSuperparentAndIndexComparator(
        this->SuperparentRounds, this->GlobalRegularIds, this->SupernodeIds);
    vtkm::cont::Algorithm::Sort(this->AttachmentIds, attachmentSuperparentAndIndexComparator);

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
    VTKM_LOG_S(vtkm::cont::LogLevel::Info,
               DebugPrint("Attachment Points Sorted on Superparent Round", __FILE__, __LINE__));
    const vtkm::Id nAttachmentDupIds = this->AttachmentIds.GetNumberOfValues();
#endif

    // Remove the duplicate values using GlobalRegularIds[AttachmentIds] for checking for equality
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::AttachmentIdsEqualComparator
      attachmentIdsEqualComparator(this->GlobalRegularIds);
    vtkm::cont::Algorithm::Unique(this->AttachmentIds, attachmentIdsEqualComparator);

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
    {
      std::stringstream debugStream;
      debugStream << "Block " << this->BlockId << ": reducing attachment point list from size "
                  << nAttachmentDupIds << " to size " << this->AttachmentIds.GetNumberOfValues()
                  << std::endl;
      VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
    }
#endif
  }

  //  2.  Set up array with bounds for subsegments
  //    We do +2 because the top level is extra, and we need an extra sentinel value at the end
  //    We initialise to NO_SUCH_ELEMENT because we may have rounds with none and we'll need to clean up serially (over the number of rounds, i.e. lg n)
  vtkm::cont::Algorithm::Copy(
    vtkm::cont::ArrayHandleConstant<vtkm::Id>(vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT,
                                              this->BaseTree->NumRounds + 2),
    this->FirstAttachmentPointInRound);

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("FirstAttachment Array resized to ") +
                          std::to_string(this->BaseTree->NumRounds + 2),
                        __FILE__,
                        __LINE__));
#endif

  // Now do a parallel set operation
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      SetFirstAttachmentPointInRoundWorklet setFirstAttachmentPointInRoundWorklet;
    this->Invoke(setFirstAttachmentPointInRoundWorklet,
                 this->AttachmentIds,     // input
                 this->SuperparentRounds, // input
                 this->FirstAttachmentPointInRound);
  }
  // The last element in the array is always set to the size as a sentinel value
  // We need to pull the firstAttachmentPointInRound array to the control environment
  // anyways for the loop afterwards so can do this set here without using Copy
  // Use regular WritePortal here since we need to update a number of values and the array should be small
  auto firstAttachmentPointInRoundPortal = this->FirstAttachmentPointInRound.WritePortal();
  firstAttachmentPointInRoundPortal.Set(this->BaseTree->NumRounds + 1,
                                        this->AttachmentIds.GetNumberOfValues());

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint("First Attachment Point Set Where Possible", __FILE__, __LINE__));
#endif
  //     Now clean up by looping through the rounds (serially - this is logarithmic at worst)
  //     We loop backwards so that the next up propagates downwards
  //     WARNING: DO NOT PARALLELISE THIS LOOP
  for (vtkm::Id roundNumber = this->BaseTree->NumRounds; roundNumber >= 0; roundNumber--)
  { // per round
    // if it still holds NSE, there are none in this round, so use the next one up
    if (vtkm::worklet::contourtree_augmented::NoSuchElement(
          firstAttachmentPointInRoundPortal.Get(roundNumber)))
    {
      firstAttachmentPointInRoundPortal.Set(roundNumber,
                                            firstAttachmentPointInRoundPortal.Get(roundNumber + 1));
    }
  } // per round

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info, DebugPrint("Subsegments Identified", __FILE__, __LINE__));
#endif
  //  3.  Initialise an array to keep track of the mapping from old supernode ID to new supernode ID
  vtkm::cont::Algorithm::Copy(
    vtkm::cont::ArrayHandleConstant<vtkm::Id>(vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT,
                                              this->BaseTree->Supernodes.GetNumberOfValues()),
    this->NewSupernodeIds);

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info, DebugPrint("Augmented Tree Prepared", __FILE__, __LINE__));
#endif
} // PrepareAugmentedTree()


// transfer of hyperstructure but not superchildren count
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::CopyHyperstructure()
{ // CopyHyperstructure()
  // we can also resize some of the additional information
  this->AugmentedTree->NumRounds = this->BaseTree->NumRounds;
  vtkm::cont::Algorithm::Copy(
    vtkm::cont::ArrayHandleConstant<vtkm::Id>(
      static_cast<vtkm::Id>(0), this->BaseTree->NumRegularNodesInRound.GetNumberOfValues()),
    this->AugmentedTree->NumRegularNodesInRound);
  vtkm::cont::Algorithm::Copy(
    vtkm::cont::ArrayHandleConstant<vtkm::Id>(
      static_cast<vtkm::Id>(0), this->BaseTree->NumSupernodesInRound.GetNumberOfValues()),
    this->AugmentedTree->NumSupernodesInRound);

  // this chunk needs to be here to prevent the HierarchicalContourTree::DebugPrint() routine from crashing
  this->AugmentedTree->FirstSupernodePerIteration.resize(
    this->BaseTree->FirstSupernodePerIteration.size());
  // this loop doesn't need to be parallelised, as it is a small size: we will fill in values later
  for (vtkm::Id roundNumber = 0;
       roundNumber < static_cast<vtkm::Id>(this->AugmentedTree->FirstSupernodePerIteration.size());
       roundNumber++)
  {
    vtkm::cont::Algorithm::Copy(
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(
        static_cast<vtkm::Id>(0),
        this->BaseTree->FirstSupernodePerIteration[roundNumber].GetNumberOfValues()),
      this->AugmentedTree->FirstSupernodePerIteration[roundNumber]);
  }

  // hyperstructure is unchanged, so we can copy it
  vtkm::cont::Algorithm::Copy(this->BaseTree->NumHypernodesInRound,
                              this->AugmentedTree->NumHypernodesInRound);
  vtkm::cont::Algorithm::Copy(this->BaseTree->NumIterations, this->AugmentedTree->NumIterations);
  this->AugmentedTree->FirstHypernodePerIteration.resize(
    this->BaseTree->FirstHypernodePerIteration.size());
  // this loop doesn't need to be parallelised, as it is a small size
  for (vtkm::Id roundNumber = 0;
       roundNumber < static_cast<vtkm::Id>(this->AugmentedTree->FirstHypernodePerIteration.size());
       roundNumber++)
  { // per round
    // duplicate the existing array
    vtkm::cont::Algorithm::Copy(this->BaseTree->FirstHypernodePerIteration[roundNumber],
                                this->AugmentedTree->FirstHypernodePerIteration[roundNumber]);
  } // per round

  // WARNING 28/05/2023:  Since this resize is for the full hyperstructure, it should be safe to put here.
  // Unless of course, anything relies on the sizes: but they were 0, so it is unlikely.
  // A search for hyperarcs.size() & hypernodes.size() in this unit confirmed that nothing uses them.
  // Nevertheless, set them all to NO_SUCH_ELEMENT out of paranoia
  // 5. Reset hypernodes, hyperarcs and superchildren using supernode IDs
  //    The hyperstructure is unchanged, but uses old supernode IDs
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->AugmentedTree->Hypernodes,                      // resize array
    this->BaseTree->Hypernodes.GetNumberOfValues(),       // new size
    vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT // set all elements to this value
  );
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->AugmentedTree->Hyperarcs,                       // resize array
    this->BaseTree->Hyperarcs.GetNumberOfValues(),        // new size
    vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT // set all elements to this value
  );
  vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
    this->AugmentedTree->Superchildren,                // resize array
    this->BaseTree->Superchildren.GetNumberOfValues(), // new size
    0                                                  // set all elements to this value
  );

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info, DebugPrint("Hyperstructure Copied", __FILE__, __LINE__));
#endif
} // CopyHyperstructure()

// reset the super IDs in the hyperstructure to the new values
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::UpdateHyperstructure(vtkm::Id roundNumber)
{ // UpdateHyperstructure()
  // now that the superstructure is known, we can find the new supernode IDs for all
  // of the old hypernodes at this level and update.Wwe want to update the entire round
  // at once, so we would like to use the firstHypernodePerIteration array.
  vtkm::Id startIndex =
    vtkm::cont::ArrayGetValue(0, this->AugmentedTree->FirstHypernodePerIteration[roundNumber]);
  vtkm::Id stopIndex = vtkm::cont::ArrayGetValue(
    vtkm::cont::ArrayGetValue(roundNumber, this->AugmentedTree->NumIterations),
    this->AugmentedTree->FirstHypernodePerIteration[roundNumber]);
  vtkm::Id selectSize = stopIndex - startIndex;
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      UpdateHyperstructureSetHyperarcsAndNodesWorklet
        updateHyperstructureSetHyperarcsAndNodesWorklet;
    // Create ArrayHandleViews of the subrange of the input and output arrays we need to process
    auto baseTreeHypernodesView =
      vtkm::cont::make_ArrayHandleView(this->BaseTree->Hypernodes, startIndex, selectSize);
    auto baseTreeHyperarcsView =
      vtkm::cont::make_ArrayHandleView(this->BaseTree->Hyperarcs, startIndex, selectSize);
    auto augmentedTreeHypernodesView =
      vtkm::cont::make_ArrayHandleView(this->AugmentedTree->Hypernodes, startIndex, selectSize);
    auto augmentedTreeHyperarcsView =
      vtkm::cont::make_ArrayHandleView(this->AugmentedTree->Hyperarcs, startIndex, selectSize);
    // Invoke the worklet with the subset view of our arrays
    this->Invoke(updateHyperstructureSetHyperarcsAndNodesWorklet,
                 baseTreeHypernodesView,      // input
                 baseTreeHyperarcsView,       // input
                 this->NewSupernodeIds,       // input
                 augmentedTreeHypernodesView, // output (the array is automatically resized here)
                 augmentedTreeHyperarcsView   // output (the array is automatically resized here)
    );
  }

  // finally, find the number of superchildren as the delta between the
  // super ID and the next hypernode's super ID
  // This is slightly tricky. Multiple supernodes may share the same hyperparent.
  {
    vtkm::Id superchildrenStartIndex =
      vtkm::cont::ArrayGetValue(0, this->AugmentedTree->FirstSupernodePerIteration[roundNumber]);
    vtkm::Id superchildrenStopIndex = vtkm::cont::ArrayGetValue(
      vtkm::cont::ArrayGetValue(roundNumber, this->AugmentedTree->NumIterations),
      this->AugmentedTree->FirstSupernodePerIteration[roundNumber]);
    vtkm::Id superchildrenSelectSize = superchildrenStopIndex - superchildrenStartIndex;
    vtkm::Id extraSelectSize = ((superchildrenStartIndex + superchildrenSelectSize) <
                                this->AugmentedTree->Hyperparents.GetNumberOfValues())
      ? superchildrenSelectSize + 1
      : superchildrenSelectSize;

    // Because we use ArrayHandleView to select the size of array for the array,
    // the index of the entry in the worklet is NOT the actual array index.
    // We need to send the starting index of supernodes into the worklet.
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      UpdateHyperstructureSetSuperchildrenWorklet updateHyperstructureSetSuperchildrenWorklet(
        this->AugmentedTree->Supernodes.GetNumberOfValues(), superchildrenStartIndex);
    // As above, we need to create views of the relevant subranges of our arrays
    auto augmentedTreeSuperarcsView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Superarcs, superchildrenStartIndex, superchildrenSelectSize);
    auto augmentedTreeHyperparentsView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Hyperparents, superchildrenStartIndex, extraSelectSize);

    this->Invoke(updateHyperstructureSetSuperchildrenWorklet,
                 this->AugmentedTree->Hypernodes,   // input
                 augmentedTreeSuperarcsView,        // input
                 augmentedTreeHyperparentsView,     // inpu
                 this->AugmentedTree->Superchildren // output
    );
  }
#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info, DebugPrint("Hyperstructure Updated", __FILE__, __LINE__));
#endif
} // UpdateHyperstructure()


// copy the remaining base level regular nodes
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::CopyBaseRegularStructure()
{ // CopyBaseRegularStructure()
  //  6.  Set up the regular node sorter for the final phase
  vtkm::cont::Algorithm::Copy(
    vtkm::cont::ArrayHandleIndex(this->AugmentedTree->RegularNodeGlobalIds.GetNumberOfValues()),
    this->AugmentedTree->RegularNodeSortOrder);
  {
    vtkm::worklet::contourtree_distributed::PermuteComparator // hierarchical_contour_tree::
      permuteComparator(this->AugmentedTree->RegularNodeGlobalIds);
    vtkm::cont::Algorithm::Sort(this->AugmentedTree->RegularNodeSortOrder, permuteComparator);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint("Regular Node Sorter Sorted", __FILE__, __LINE__));
#endif

  //  7.  Cleanup at level = 0.  The principal task here is to insert all of the regular
  //      nodes in the original block into the regular arrays. The problem is that we
  //      haven't got a canonical list of them since in the original hierarchical tree,
  //      they might have been passed upwards as part of the boundary resolution.  We
  //      now have a choice: we can take all "unfiled" regular nodes in the original hierarchical
  //      tree, or we can return to the block of data. The difference here is that the
  //      "unfiled" regular nodes can include nodes from other blocks which were passed up
  //      and retained in both partners.  On the other hand, we already have the list of
  //      unfiled regular nodes, so the overhead for using them is not huge. And if we
  //      return to the block, we will need to pass it in as a parameter and template
  //      on mesh type.  So, for the purposes of tidy coding, I shall use the first
  //      option, which means that not all of the Level 0 regular nodes belong to the block.
  {
    //    For each regular node, if it hasn't been transferred to the new tree, search for the superarc to which it belongs
    //    default the superparent to NO_SUCH_ELEMENT to use as a flag for "we can ignore it"
    //    now loop, finding the superparent for each node needed and set the approbriate value or set to
    //    NO_SUCH_ELEMENT if not needed. The worklet also automatically sized our arrays
    // temporary array so we can stream compact (aka CopyIf) afterwards
    vtkm::worklet::contourtree_augmented::IdArrayType tempRegularNodesNeeded;
    // create the worklet
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      FindSuperparentForNecessaryNodesWorklet findSuperparentForNecessaryNodesWorklet(
        this->MeshBlockOrigin, this->MeshBlockSize, this->MeshGlobalSize);
    // Get a FindRegularByGlobal and FindSuperArcForUnknownNode execution object for our worklet
    auto findRegularByGlobal = this->AugmentedTree->GetFindRegularByGlobal();
    auto findSuperArcForUnknownNode = this->AugmentedTree->GetFindSuperArcForUnknownNode();

    // execute the worklet
    this->Invoke(findSuperparentForNecessaryNodesWorklet, // the worklet to call
                 // inputs
                 this->BaseTree->RegularNodeGlobalIds, // input domain
                 this->BaseTree->Superparents,         // input
                 this->BaseTree->DataValues,           // input
                 this->BaseTree->Superarcs,            // input
                 this->NewSupernodeIds,                // input
                 // Execution objects from the AugmentedTree
                 findRegularByGlobal,
                 findSuperArcForUnknownNode,
                 // Output arrays to populate
                 this->RegularSuperparents, // output
                 tempRegularNodesNeeded     // output. will be CopyIf'd to this->RegularNodesNeeded
    );

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
    {
      std::stringstream debugStream;
      VTKM_LOG_S(vtkm::cont::LogLevel::Info,
                 DebugPrint("Regular Node Superparents Found", __FILE__, __LINE__));
      vtkm::worklet::contourtree_augmented::PrintHeader(tempRegularNodesNeeded.GetNumberOfValues(),
                                                        debugStream);
      vtkm::worklet::contourtree_augmented::PrintIndices(
        "RegularNodesNeeded", tempRegularNodesNeeded, -1, debugStream);
      VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
    }
#endif

    // We now compress to get the set of nodes to transfer. I.e., remove all
    // NO_SUCH_ELEMENT entires and copy the values to keep to our proper arrays
    vtkm::worklet::contourtree_augmented::NotNoSuchElementPredicate notNoSuchElementPredicate;
    vtkm::cont::Algorithm::CopyIf(
      tempRegularNodesNeeded,   // input data
      tempRegularNodesNeeded,   // stencil (same as input)
      this->RegularNodesNeeded, // target array (will be resized)
      notNoSuchElementPredicate // predicate returning true of element is NOT NO_SUCH_ELEMENT
    );
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint("Regular Node List Compressed", __FILE__, __LINE__));
#endif

  // resize the regular arrays to fit
  vtkm::Id numRegNeeded = this->RegularNodesNeeded.GetNumberOfValues();
  vtkm::Id numExistingRegular = this->AugmentedTree->RegularNodeGlobalIds.GetNumberOfValues();
  vtkm::Id numTotalRegular = numExistingRegular + numRegNeeded;
  {
    // Resize the array, while preserving the orginial values and initalizing new values
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->AugmentedTree->RegularNodeGlobalIds, numTotalRegular, static_cast<vtkm::Id>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector<FieldType>(
      this->AugmentedTree->DataValues, numTotalRegular, static_cast<FieldType>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->AugmentedTree->RegularNodeSortOrder, numTotalRegular, static_cast<vtkm::Id>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->AugmentedTree->Regular2Supernode,
      numTotalRegular,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->AugmentedTree->Superparents, numTotalRegular, static_cast<vtkm::Id>(0));
  }

  // OK:  we have a complete list of the nodes to transfer. Since we make no guarantees (yet) about sorting, they just copy across
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::CopyBaseRegularStructureWorklet
      copyBaseRegularStructureWorklet(numExistingRegular);
    // NOTE: We require the input arrays (aside form the input domain) to be permutted by the
    //       regularNodesNeeded input domain so that we can use FieldIn instead of WholeArrayIn
    // NOTE: We require ArrayHandleView for the output arrays of the range [numExistingRegular:end] so
    //       that we can use FieldOut instead of requiring WholeArrayInOut
    // input domain for the worklet of [0, regularNodesNeeded.GetNumberOfValues()]
    auto regularNodesNeededRange =
      vtkm::cont::ArrayHandleIndex(this->RegularNodesNeeded.GetNumberOfValues());
    // input baseTree->regularNodeGlobalIds permuted by regularNodesNeeded
    auto baseTreeRegularNodeGlobalIdsPermuted = vtkm::cont::make_ArrayHandlePermutation(
      this->RegularNodesNeeded, this->BaseTree->RegularNodeGlobalIds);
    // input baseTree->dataValues permuted by regularNodesNeeded
    auto baseTreeDataValuesPermuted =
      vtkm::cont::make_ArrayHandlePermutation(this->RegularNodesNeeded, this->BaseTree->DataValues);
    // input regularSuperparents permuted by regularNodesNeeded
    auto regularSuperparentsPermuted =
      vtkm::cont::make_ArrayHandlePermutation(this->RegularNodesNeeded, this->RegularSuperparents);
    // input view of augmentedTree->regularNodeGlobalIds[numExistingRegular:]
    auto augmentedTreeRegularNodeGlobalIdsView =
      vtkm::cont::make_ArrayHandleView(this->AugmentedTree->RegularNodeGlobalIds,
                                       numExistingRegular, // start writing at numExistingRegular
                                       numRegNeeded);      // fill until the end
    // input view of augmentedTree->dataValues[numExistingRegular:]
    auto augmentedTreeDataValuesView =
      vtkm::cont::make_ArrayHandleView(this->AugmentedTree->DataValues,
                                       numExistingRegular, // start writing at numExistingRegular
                                       numRegNeeded);      // fill until the end
    // input view of augmentedTree->superparents[numExistingRegular:]
    auto augmentedTreeSuperparentsView =
      vtkm::cont::make_ArrayHandleView(this->AugmentedTree->Superparents,
                                       numExistingRegular, // start writing at numExistingRegular
                                       numRegNeeded);      // fill until the end
    // input view of  augmentedTree->regularNodeSortOrder[numExistingRegular:]
    auto augmentedTreeRegularNodeSortOrderView =
      vtkm::cont::make_ArrayHandleView(this->AugmentedTree->RegularNodeSortOrder,
                                       numExistingRegular, // start writing at numExistingRegular
                                       numRegNeeded);      // fill until the end
    this->Invoke(copyBaseRegularStructureWorklet,          // worklet to call
                 regularNodesNeededRange,                  // input domain
                 baseTreeRegularNodeGlobalIdsPermuted,     // input
                 baseTreeDataValuesPermuted,               // input
                 regularSuperparentsPermuted,              // input
                 augmentedTreeRegularNodeGlobalIdsView,    // output
                 augmentedTreeDataValuesView,              // output
                 augmentedTreeSuperparentsView,            // output
                 augmentedTreeRegularNodeSortOrderView     // output
    );
  }

  // Reset the number of regular nodes in round 0
  /* ----- MINGZHE: DEBUG ----- */
  vtkm::Id regularNodesInRound0 =
    this->AugmentedTree->NumRegularNodesInRound.ReadPortal().Get(0) + numRegNeeded;
  this->AugmentedTree->NumRegularNodesInRound.WritePortal().Set(0, regularNodesInRound0);

  //  Finally, we resort the regular node sort order
  {
    vtkm::worklet::contourtree_distributed::PermuteComparator // hierarchical_contour_tree::
      permuteComparator(this->AugmentedTree->RegularNodeGlobalIds);
    vtkm::cont::Algorithm::Sort(this->AugmentedTree->RegularNodeSortOrder, permuteComparator);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint("Base Regular Structure Copied", __FILE__, __LINE__));
#endif

} // CopyBaseRegularStructure()


// subroutines for CopySuperstructure
// gets a list of all the old supernodes to transfer at this level (ie except attachment points
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::RetrieveOldSupernodes(vtkm::Id roundNumber)
{ // RetrieveOldSupernodes()
  //  a.  Transfer supernodes from same level of old tree minus attachment points, storing by global regular ID not regular ID
  //     Use compression to get the set of supernode IDs that we want to keep
  // Previously, it made the hard assumption that all attachment points were transferred & used that to suppress
  // them. Now it can do that no longer, which gives us a choice.  We can pass in the threshold & volume array
  // and test here, but that's not ideal as it does the same test in multiple places. Alternatively, we can
  // look up whether the supernode is already present in the structure, which has an associated search cost.
  // BUT, we have an array called newSupernodeIDs for the purpose already, so that's how we'll do it.

  vtkm::Id supernodeIndexBase =
    vtkm::cont::ArrayGetValue(0, this->BaseTree->FirstSupernodePerIteration[roundNumber]);
  vtkm::cont::ArrayHandleCounting<vtkm::Id> supernodeIdVals(
    supernodeIndexBase, // start
    1,                  // step
    this->BaseTree->NumSupernodesInRound.ReadPortal().Get(roundNumber));

  {
    // Reset this-KeptSupernodes to the right size and initalize with NO_SUCH_ELEMENT.
    vtkm::cont::Algorithm::Copy(
      // Create const array to copy
      vtkm::cont::ArrayHandleConstant<vtkm::Id>(
        vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT,
        this->BaseTree->NumSupernodesInRound.ReadPortal().Get(roundNumber)),
      // target array
      this->KeptSupernodes);

    // Create the predicate for the CopyIf
    vtkm::worklet::contourtree_augmented::NoSuchElementPredicate NoSuchElementPredicate;
    // Copy supernodeId to this->KeptSupernodes
    vtkm::cont::Algorithm::CopyIf(
      // first we generate a list of supernodeIds
      supernodeIdVals,
      // Stencil with baseTree->superarcs[supernodeID]
      vtkm::cont::make_ArrayHandleView(
        this->NewSupernodeIds, supernodeIndexBase, this->KeptSupernodes.GetNumberOfValues()),
      // And the CopyIf compresses the array to eliminate unnecssary elements
      // save to this->KeptSupernodes
      this->KeptSupernodes,
      // then our predicate identifies all necessary points. These
      // are all points that suffice the condition
      // vtkm::Id supernodeID = keptSupernode + supernodeIndexBase;
      // !noSuchElement(baseTree->superarcs[supernodeID]);
      NoSuchElementPredicate);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Old Supernodes Retrieved"),
                        __FILE__,
                        __LINE__));
#endif
} // RetrieveOldSupernodes()


// resizes the arrays for the level
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::ResizeArrays(vtkm::Id roundNumber)
{ // ResizeArrays()
  // at this point, we know how many supernodes are kept from the same level of the old tree
  // we can also find out how many supernodes are being inserted, which gives us the correct amount to expand by, saving a double resize() call
  // note that some of these arrays could probably be resized later, but it's cleaner this way
  // also note that if it becomes a problem, we could resize all of the arrays to baseTree->supernodes.size() + # of attachmentPoints as an over-estimate
  // then cut them down at the end.  The code would however be even messier if we did so
  vtkm::Id numSupernodesAlready = this->AugmentedTree->Supernodes.GetNumberOfValues();
  vtkm::Id numInsertedSupernodes =
    vtkm::cont::ArrayGetValue(roundNumber + 1, this->FirstAttachmentPointInRound) -
    vtkm::cont::ArrayGetValue(roundNumber, this->FirstAttachmentPointInRound);
  vtkm::Id numSupernodesThisLevel =
    numInsertedSupernodes + this->KeptSupernodes.GetNumberOfValues();
  vtkm::Id newSupernodeCount = numSupernodesAlready + numSupernodesThisLevel;

  // conveniently, the value numSupernodesThisLevel is the number of supernodes *!AND!* regular nodes to store for the round
  vtkm::worklet::contourtree_augmented::IdArraySetValue(
    roundNumber, numSupernodesThisLevel, this->AugmentedTree->NumRegularNodesInRound);
  vtkm::worklet::contourtree_augmented::IdArraySetValue(
    roundNumber, numSupernodesThisLevel, this->AugmentedTree->NumSupernodesInRound);
  vtkm::worklet::contourtree_augmented::IdArraySetValue(
    0, numSupernodesAlready, this->AugmentedTree->FirstSupernodePerIteration[roundNumber]);

  // resize the arrays accordingly.
  // NOTE: We have to resize arrays (not just allocate them) as we need to
  // expand the arrays while preserving the original values
  {
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Supernodes,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Supernodes,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Superarcs,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Hyperparents,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Super2Hypernode,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->WhichRound,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->WhichIteration,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);

    // we also know that only supernodes are needed as regular nodes at each level, so we resize those here as well
    // we note that it might be possible to update all regular IDs at the end, but leave that optimisation out for now
    // therefore we resize the regular-sized arrays as well
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->RegularNodeGlobalIds,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector<FieldType>(
      this->AugmentedTree->DataValues, newSupernodeCount, static_cast<FieldType>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Regular2Supernode,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
    vtkm::worklet::contourtree_augmented::ResizeVector(
      this->AugmentedTree->Superparents,
      newSupernodeCount,
      vtkm::worklet::contourtree_augmented::NO_SUCH_ELEMENT);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(
    vtkm::cont::LogLevel::Info,
    DebugPrint(std::string("Round ") + std::to_string(roundNumber) + std::string(" Arrays Resized"),
               __FILE__,
               __LINE__));
#endif

  // The next task is to assemble a sorting array which we will use to construct the new superarcs, containing both the
  // kept supernodes and the attachment points.  The attachment points are easier since we already have them, so we start
  // by allocating space and copying them in: this means another set of arrays for the individual elements.  However, we do not
  // need all of the data elements, since superparentRound is fixed (and equal to roundNumber inside this loop), and whichRound will be reset
  vtkm::cont::Algorithm::Copy(vtkm::cont::ArrayHandleIndex(numSupernodesThisLevel),
                              this->SupernodeSorter);
  {
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->GlobalRegularIdSet, numSupernodesThisLevel, static_cast<vtkm::Id>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector<FieldType>(
      this->DataValueSet, numSupernodesThisLevel, static_cast<FieldType>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->SuperparentSet, numSupernodesThisLevel, static_cast<vtkm::Id>(0));
    vtkm::worklet::contourtree_augmented::ResizeVector<vtkm::Id>(
      this->SupernodeIdSet, numSupernodesThisLevel, static_cast<vtkm::Id>(0));
  }
#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Sorter Set Resized"),
                        __FILE__,
                        __LINE__));
#endif

  // b. Transfer attachment points for level into new supernode array
  // NOTE: this means the set of attachment points that we have determined by swapping
  //       need to be inserted onto a superarc at this level. All of them should be from
  //       lower levels originally, but are being moved up to this level for insertion
  // to copy them in, we use the existing array of attachment point IDs by round
  {
    vtkm::Id firstAttachmentPointInRoundCurrent =
      vtkm::cont::ArrayGetValue(roundNumber, this->FirstAttachmentPointInRound);
    vtkm::Id firstAttachmentPointInRoundNext =
      vtkm::cont::ArrayGetValue(roundNumber + 1, this->FirstAttachmentPointInRound);
    vtkm::Id currRange = firstAttachmentPointInRoundNext - firstAttachmentPointInRoundCurrent;
    auto attachmentPointIdView =
      vtkm::cont::make_ArrayHandleView(this->AttachmentIds,                // array to subset
                                       firstAttachmentPointInRoundCurrent, // start index
                                       currRange);                         // count
    // Permute the source arrays for the copy
    auto globalRegularIdsPermuted =
      vtkm::cont::make_ArrayHandlePermutation(attachmentPointIdView, // index array
                                              this->GlobalRegularIds // value array
      );
    auto dataValuesPermuted =
      vtkm::cont::make_ArrayHandlePermutation(attachmentPointIdView, this->DataValues);
    auto superparentsPermuted =
      vtkm::cont::make_ArrayHandlePermutation(attachmentPointIdView, this->Superparents);
    auto supernodeIdsPermuted =
      vtkm::cont::make_ArrayHandlePermutation(attachmentPointIdView, this->SupernodeIds);
    // Now use CopySubRange to copy the values into the right places. This allows
    // us to place them in the right place and avoid shrinking the array on Copy
    vtkm::cont::Algorithm::CopySubRange(
      // copy all values of our permutted array
      globalRegularIdsPermuted,
      0,
      globalRegularIdsPermuted.GetNumberOfValues(),
      // copy target
      this->GlobalRegularIdSet,
      0);
    vtkm::cont::Algorithm::CopySubRange(
      dataValuesPermuted, 0, dataValuesPermuted.GetNumberOfValues(), this->DataValueSet, 0);
    vtkm::cont::Algorithm::CopySubRange(
      superparentsPermuted, 0, superparentsPermuted.GetNumberOfValues(), this->SuperparentSet, 0);
    vtkm::cont::Algorithm::CopySubRange(
      supernodeIdsPermuted, 0, supernodeIdsPermuted.GetNumberOfValues(), this->SupernodeIdSet, 0);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Attachment Points Transferred"),
                        __FILE__,
                        __LINE__));
#endif

  // Now we copy in the kept supernodes: this used to mean only the non-attachment points
  // now it includes the attachment points at this level that the simplification removed
  // so they need to be put back where they were
  // However, that means that all of them do exist in the base tree, so we can copy from there
  {
    auto oldRegularIdArr =
      vtkm::cont::make_ArrayHandlePermutation(this->KeptSupernodes,        // index
                                              this->BaseTree->Supernodes); // values
    // Permute the source arrays for the copy
    auto baseTreeregularNodeGlobalIdsPermuted = vtkm::cont::make_ArrayHandlePermutation(
      oldRegularIdArr, this->BaseTree->RegularNodeGlobalIds);
    auto baseTreeDataValuesPermuted =
      vtkm::cont::make_ArrayHandlePermutation(oldRegularIdArr, this->BaseTree->DataValues);

    // Now use CopySubRange to copy the values into the right places. This allows
    // us to place them in the right place and avoids shrinking the array on Copy
    vtkm::cont::Algorithm::CopySubRange(baseTreeregularNodeGlobalIdsPermuted,
                                        0,
                                        baseTreeregularNodeGlobalIdsPermuted.GetNumberOfValues(),
                                        this->GlobalRegularIdSet,
                                        numInsertedSupernodes);
    vtkm::cont::Algorithm::CopySubRange(baseTreeDataValuesPermuted,
                                        0,
                                        baseTreeDataValuesPermuted.GetNumberOfValues(),
                                        this->DataValueSet,
                                        numInsertedSupernodes);
    vtkm::cont::Algorithm::CopySubRange(this->KeptSupernodes,
                                        0,
                                        this->KeptSupernodes.GetNumberOfValues(),
                                        this->SupernodeIdSet,
                                        numInsertedSupernodes);
    // For this->SuperparentSet we need to set values to
    // superparentSet[supernodeSetID]  = oldSupernodeID | (isAscending(baseTree->superarcs[oldSupernodeID]) ? IS_ASCENDING: 0x00);
    // so we use an ArrayHanldeDecorator instead to compute the values and copy them in place
    auto setSuperparentSetArrayDecorator = vtkm::cont::make_ArrayHandleDecorator(
      this->KeptSupernodes.GetNumberOfValues(),
      vtkm::worklet::contourtree_distributed::hierarchical_augmenter::SetSuperparentSetDecorator{},
      this->KeptSupernodes,
      this->BaseTree->Superarcs);
    vtkm::cont::Algorithm::CopySubRange(setSuperparentSetArrayDecorator,
                                        0,
                                        this->KeptSupernodes.GetNumberOfValues(),
                                        this->SuperparentSet,
                                        numInsertedSupernodes);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Kept Supernodes Transferred"),
                        __FILE__,
                        __LINE__));
#endif

  //  c.  Create a permutation array and sort supernode segment by a. superparent, b. value, d. global index to establish segments (reversing as needed)
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      AttachmentAndSupernodeComparator<FieldType>
        attachmentAndSupernodeComparator(
          this->SuperparentSet, this->DataValueSet, this->GlobalRegularIdSet);
    vtkm::cont::Algorithm::Sort(this->SupernodeSorter, attachmentAndSupernodeComparator);
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Sorter Set Sorted"),
                        __FILE__,
                        __LINE__));
#endif

  //  d.  Build the inverse permutation array for lookup purposes:
  {
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      ResizeArraysBuildNewSupernodeIdsWorklet resizeArraysBuildNewSupernodeIdsWorklet(
        numSupernodesAlready);
    auto supernodeIndex = vtkm::cont::ArrayHandleIndex(this->SupernodeSorter.GetNumberOfValues());
    // auto supernodeIdSetPermuted =
    //  vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->SupernodeIdSet);
    auto globalRegularIdSetPermuted =
      vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->GlobalRegularIdSet);
    auto findRegularByGlobal = this->BaseTree->GetFindRegularByGlobal();
    this->Invoke(
      resizeArraysBuildNewSupernodeIdsWorklet,
      supernodeIndex, // input domain. We only need the index because supernodeIdSetPermuted already does the permute
      // supernodeIdSetPermuted, // input input supernodeIDSet permuted by supernodeSorter to allow for FieldIn
      globalRegularIdSetPermuted,
      findRegularByGlobal,
      this->BaseTree->Regular2Supernode,
      this
        ->NewSupernodeIds // output/input (both are necessary since not all values will be overwritten)
    );

    // Add const ExecObjectType1& findRegularByGlobal,
    // Add baseTree->regular2supernode
  }

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Sorting Arrays Built"),
                        __FILE__,
                        __LINE__));
#endif
} // ResizeArrays()


// adds a round full of superarcs (and regular nodes) to the tree
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::CreateSuperarcs(vtkm::Id roundNumber)
{ // CreateSuperarcs()
  // retrieve the ID number of the first supernode at this level
#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Starting CreateSuperarcs()"),
                        __FILE__,
                        __LINE__));
#endif

  vtkm::Id currNumIterations =
    vtkm::cont::ArrayGetValue(roundNumber, this->AugmentedTree->NumIterations);
  vtkm::Id numSupernodesAlready =
    vtkm::cont::ArrayGetValue(0, this->AugmentedTree->FirstSupernodePerIteration[roundNumber]);

  // e.	Connect superarcs for the level & set hyperparents & superchildren count, whichRound, whichIteration, super2hypernode
  // 24/05/2023: Expansion of comment to help debug.
  // At this point, we know that all higher rounds are correctly constructed, and that any attachment points that survived simplification
  // have already been inserted in a higher round.

  // The sort should have resulted in the supernodes being segmented along old superarcs.  Most supernodes should be in a segment of length
  // 1, and should be their own superparent in the sort array.  But we can't readily test that, because other supernodes may also have them
  // as the superparent.

  // This loop will principally determine the superarc for each supernode.  For this, the rules break down to:
  // 1.	If the supernode is the global root, connect it nowhere
  // 2.	If the supernode is the last in the set of all supernodes in this round, treat it as the end of a segment
  // 3. If the supernode is the last in a segment by superarc, connect it to the target of its superparent in the old tree, using the new supernode ID
  // 4.	Otherwise, connect to the new supernode ID of the next supernode in the segment

  // In each case, we will need to preserve the ascending / descending flag

  // We will also have to set the first supernode per iteration - if possible, in a separate loop

  // We need this to determine which supernodes are inserted and which are attached (see below)
  vtkm::Id numInsertedSupernodes =
    (vtkm::cont::ArrayGetValue(roundNumber + 1, this->FirstAttachmentPointInRound) -
     vtkm::cont::ArrayGetValue(roundNumber, this->FirstAttachmentPointInRound));

  { // START Call CreateSuperarcsWorklet (scope to delete temporary variables)
    // create the worklet
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::CreateSuperarcsWorklet<
      FieldType>
      createSuperarcsWorklet(
        numSupernodesAlready, this->BaseTree->NumRounds, numInsertedSupernodes, roundNumber);

    // create fancy arrays needed to allow use of FieldIn for worklet parameters

    // permutedSupernodeIdSet may be NO_SUCH_ELEMENT if not already in the base tree
    // We cannot use it for permutation index
    // For any array using permutedSupernodeIdSet as indices, we put them into the data exec object.
    auto permutedSupernodeIdSet =
      vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->SupernodeIdSet);

    auto permutedGlobalRegularIdSet =
      vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->GlobalRegularIdSet);
    auto permutedDataValueSet =
      vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->DataValueSet);

    // Create a view of the range of this->AugmentedTree->Superarcs that will be updated by the worklet
    // so that we can use FieldOut as the type in the worklet
    auto augmentedTreeSuperarcsView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Superarcs,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );

    auto augmentedTreeHyperparentsView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Hyperparents,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeSuper2HypernodeView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Super2Hypernode,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeWhichRoundView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->WhichRound,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeWhichIterationView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->WhichIteration,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeRegularNodeGlobalIdsView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->RegularNodeGlobalIds,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeDataValuesView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->DataValues,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeRegular2SupernodeView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Regular2Supernode,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );
    auto augmentedTreeSuperparentsView = vtkm::cont::make_ArrayHandleView(
      this->AugmentedTree->Superparents,
      numSupernodesAlready,                     // start view here
      this->SupernodeSorter.GetNumberOfValues() // select this many values
    );

    // Required execution objects to call other functions
    auto findSuperArcForUnknownNode = this->AugmentedTree->GetFindSuperArcForUnknownNode();

    // Execution object used to encapsulate data form the BaseTree to avoid the limit of 20 input parameters per worklet
    auto createSuperarcsDataExecObj =
      vtkm::worklet::contourtree_distributed::hierarchical_augmenter::CreateSuperarcsDataExec(
        this->BaseTree->Hyperparents,
        this->BaseTree->WhichRound,
        this->BaseTree->WhichIteration,
        this->BaseTree->Supernodes,
        this->BaseTree->Superarcs,
        this->BaseTree->Superparents,
        this->BaseTree->Super2Hypernode,
        this->BaseTree->Hypernodes,
        this->SuperparentSet,
        this->NewSupernodeIds);

    // invoke the worklet
    this->Invoke(createSuperarcsWorklet, // the worklet
                 // inputs
                 this->SupernodeSorter,      // input domain (WholeArrayIn)
                 permutedSupernodeIdSet,     // input (FieldIn)
                 permutedGlobalRegularIdSet, // input (FieldIn)
                 permutedDataValueSet,       // input (FieldIn)
                 findSuperArcForUnknownNode, // input (Execution object)
                 createSuperarcsDataExecObj, // input (Execution object with BaseTreeData
                 // Outputs
                 this->AugmentedTree->Supernodes,       // input/output
                 augmentedTreeSuperarcsView,            // output
                 augmentedTreeHyperparentsView,         // output
                 augmentedTreeSuper2HypernodeView,      // output
                 augmentedTreeWhichRoundView,           // output
                 augmentedTreeWhichIterationView,       // output
                 augmentedTreeRegularNodeGlobalIdsView, // output
                 augmentedTreeDataValuesView,           // output
                 augmentedTreeRegular2SupernodeView,    // output
                 augmentedTreeSuperparentsView          // output
    );
  } // END Call CreateSuperarcsWorklet

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Details Filled in For Supernodes "),
                        __FILE__,
                        __LINE__));
#endif

  // Now, in order to set the first supernode per iteration, we do an additional loop
  // We are guaranteed that all supernodes at this level are implicitly sorted by iteration, so we test for ends of segments
  // NOTE that we do this after the previous loop, since we depend on a value that it has set
  {
    vtkm::cont::ArrayHandleCounting<vtkm::Id> tempIndex(1, 1, currNumIterations - 1);
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      CreateSuperarcsSetFirstSupernodePerIterationWorklet
        createSuperarcsSetFirstSupernodePerIterationWorklet(numSupernodesAlready);
    vtkm::cont::ArrayHandleIndex tempSupernodeIndex(this->SupernodeSorter.GetNumberOfValues());
    this->Invoke(createSuperarcsSetFirstSupernodePerIterationWorklet,
                 tempSupernodeIndex,
                 this->AugmentedTree->WhichIteration,                         // input
                 this->AugmentedTree->FirstSupernodePerIteration[roundNumber] // input/output
    );
  }

  // since there's an extra entry in the firstSupernode array as a sentinel, set it
  vtkm::worklet::contourtree_augmented::IdArraySetValue(
    currNumIterations,                                           // index
    this->AugmentedTree->Supernodes.GetNumberOfValues(),         // new value
    this->AugmentedTree->FirstSupernodePerIteration[roundNumber] // array
  );
  // The following loop should be safe in parallel since there should never be two zeros in sequence, i.e., the next
  // entry after a zero will always be valid, regardless of execution order
  // This was added because in rare cases there are no supernodes transferred in an iteration, for example because there
  // are no available upper leaves to prune. If this is case, we are guaranteed that there will be available lower leaves
  // so the next iteration will have a non-zero number.  We had a major bug from this, and it's cropped back up in the
  // Hierarchical Augmentation, so I'm expanding the comment just in case.
  {
    vtkm::cont::ArrayHandleCounting<vtkm::Id> tempIndex(1, 1, currNumIterations - 1);
    vtkm::worklet::contourtree_distributed::hierarchical_augmenter::
      CreateSuperarcsUpdateFirstSupernodePerIterationWorklet
        createSuperarcsUpdateFirstSupernodePerIterationWorklet;
    this->Invoke(createSuperarcsUpdateFirstSupernodePerIterationWorklet,
                 tempIndex,                                                   // input index
                 this->AugmentedTree->FirstSupernodePerIteration[roundNumber] // input/output
    );
  }

  // We have one last bit of cleanup to do.  If there were attachment points,
  // then the round in which they transfer has been removed
  // While it is possible to turn this into a null round, it is better to
  // reduce the iteration count by one and resize the arrays
  // To do this, we access the *LAST* element written and check to see whether
  // it is in the final iteration (according to the base tree)
  // But there might be *NO* supernodes in the round, so we check first
  if (currNumIterations > 0)
  { // at least one iteration
    vtkm::Id lastSupernodeThisLevel = this->AugmentedTree->Supernodes.GetNumberOfValues() - 1;
    vtkm::Id lastIterationThisLevel = vtkm::worklet::contourtree_augmented::MaskedIndex(
      vtkm::cont::ArrayGetValue(lastSupernodeThisLevel, this->AugmentedTree->WhichIteration));
    // if there were no attachment points, it will be in the last iteration: if there were
    // attachment points, it will be in the previous one
    if (lastIterationThisLevel < currNumIterations - 1)
    { // attachment point round was removed
      // decrement the iteration count (still with an extra element as sentinel)
      vtkm::Id iterationArraySize = currNumIterations;
      // decrease iterations by 1. I.e,: augmentedTree->nIterations[roundNo]--;
      vtkm::worklet::contourtree_augmented::IdArraySetValue(
        roundNumber,                       // index
        currNumIterations - 1,             // new value
        this->AugmentedTree->NumIterations // array
      );
      // shrink the supernode array
      this->AugmentedTree->FirstSupernodePerIteration[roundNumber].Allocate(
        iterationArraySize, vtkm::CopyFlag::On); // shrink array but keep values
      vtkm::worklet::contourtree_augmented::IdArraySetValue(
        iterationArraySize - 1,                                      // index
        this->AugmentedTree->Supernodes.GetNumberOfValues(),         // new value
        this->AugmentedTree->FirstSupernodePerIteration[roundNumber] // array
      );

      // for the hypernode array, the last iteration is guaranteed not to have hyperarcs by construction
      // so the last iteration will already have the correct sentinel value, and we just need to shrink the array
      this->AugmentedTree->FirstHypernodePerIteration[roundNumber].Allocate(
        iterationArraySize, vtkm::CopyFlag::On); // shrink array but keep values
    }                                            // attachment point round was removed
  }                                              // at least one iteration

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
  VTKM_LOG_S(vtkm::cont::LogLevel::Info,
             DebugPrint(std::string("Round ") + std::to_string(roundNumber) +
                          std::string(" Superarcs Created "),
                        __FILE__,
                        __LINE__));
#endif

  // in the interests of debug, we resize the sorting array to zero here,
  // even though we will re-resize them in the next function
  this->SupernodeSorter.ReleaseResources();
  this->GlobalRegularIdSet.ReleaseResources();
  this->DataValueSet.ReleaseResources();
  this->SuperparentSet.ReleaseResources();
  this->SupernodeIdSet.ReleaseResources();
} // CreateSuperarcs()


// debug routine
template <typename FieldType>
std::string HierarchicalAugmenter<FieldType>::DebugPrint(std::string message,
                                                         const char* fileName,
                                                         long lineNum)
{ // DebugPrint()
  std::stringstream resultStream;
  resultStream << "%" << std::endl;
  resultStream << "----------------------------------------" << std::endl;
  resultStream << std::setw(30) << std::left << fileName << ":" << std::right << std::setw(4)
               << lineNum << std::endl;
  resultStream << "Block " << std::setw(4) << this->BlockId << ": " << std::left << message
               << std::endl;
  resultStream << "----------------------------------------" << std::endl;

#ifdef DEBUG_PRINT_HIERARCHICAL_CONTOUR_TREE
  resultStream << this->BaseTree->DebugPrint(
    (message + std::string(" Base Tree")).c_str(), fileName, lineNum);
  resultStream << this->AugmentedTree->DebugPrint(
    (message + std::string(" Augmented Tree")).c_str(), fileName, lineNum);
#endif
  resultStream << "========================================" << std::endl;
  resultStream << "Local List of Attachment Points" << std::endl;
  vtkm::worklet::contourtree_augmented::PrintHeader(this->GlobalRegularIds.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Global Regular Ids", this->GlobalRegularIds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintValues(
    "Data Values", this->DataValues, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Supernode Ids", this->SupernodeIds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Superparents", this->Superparents, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Superparent Rounds", this->SuperparentRounds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "WhichRounds", this->WhichRounds, -1, resultStream);
  resultStream << std::endl;
  resultStream << "Outgoing Attachment Points" << std::endl;
  vtkm::worklet::contourtree_augmented::PrintHeader(
    this->OutData.GlobalRegularIds.GetNumberOfValues(), resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Global Regular Ids", this->OutData.GlobalRegularIds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintValues(
    "Out Data Values", this->OutData.DataValues, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Supernode Ids", this->OutData.SupernodeIds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Superparents", this->OutData.Superparents, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Superparent Rounds", this->OutData.SuperparentRounds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out WhichRounds", this->OutData.WhichRounds, -1, resultStream);
  resultStream << std::endl;
  // we only output the incoming attachment points in the data exchange debug print
  if (message.find(std::string("In Attachment Points Received")) != std::string::npos)
  {
    resultStream << "Incoming Attachment Points" << std::endl;
    vtkm::worklet::contourtree_augmented::PrintHeader(
      this->InData.GlobalRegularIds.GetNumberOfValues(), resultStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "In Global Regular Ids", this->InData.GlobalRegularIds, -1, resultStream);
    vtkm::worklet::contourtree_augmented::PrintValues(
      "In Data Values", this->InData.DataValues, -1, resultStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "In Supernode Ids", this->InData.SupernodeIds, -1, resultStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "In Superparents", this->InData.Superparents, -1, resultStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "In Superparent Rounds", this->InData.SuperparentRounds, -1, resultStream);
    vtkm::worklet::contourtree_augmented::PrintIndices(
      "In WhichRounds", this->InData.WhichRounds, -1, resultStream);
    resultStream << std::endl;
  }
  resultStream << "Holding Arrays" << std::endl;
  vtkm::worklet::contourtree_augmented::PrintHeader(
    this->FirstAttachmentPointInRound.GetNumberOfValues(), resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "First Attach / Rd", this->FirstAttachmentPointInRound, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintHeader(this->AttachmentIds.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "AttachmentIds", this->AttachmentIds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintHeader(this->NewSupernodeIds.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "New Supernode Ids", this->NewSupernodeIds, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintHeader(this->KeptSupernodes.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Kept Supernodes", this->KeptSupernodes, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintHeader(this->SupernodeSorter.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Supernode Sorter", this->SupernodeSorter, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Global Regular Id", this->GlobalRegularIdSet, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintValues(
    "Data Values", this->DataValueSet, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Superparents", this->SuperparentSet, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "SupernodeIds", this->SupernodeIdSet, -1, resultStream);
  resultStream << std::endl;
  resultStream << std::endl;

  vtkm::worklet::contourtree_augmented::PrintHeader(this->SupernodeSorter.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Supernode Id", this->SupernodeSorter, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Superparent",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->SuperparentSet),
    -1,
    resultStream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Value",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->DataValueSet),
    -1,
    resultStream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Global Id",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->GlobalRegularIdSet),
    -1,
    resultStream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Supernode Id",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->SupernodeIdSet),
    -1,
    resultStream);
  resultStream << std::endl;
  resultStream << std::endl;

  vtkm::worklet::contourtree_augmented::PrintHeader(this->RegularSuperparents.GetNumberOfValues(),
                                                    resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "RegularNodesNeeded", this->RegularNodesNeeded, -1, resultStream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "RegularSuperparents", this->RegularSuperparents, -1, resultStream);
  resultStream << std::endl;
  // for now, nothing here at all
  //std::string hierarchicalTreeDotString = HierarchicalContourTreeDotGraphPrint(message,
  //                             this->BaseTree,
  //                             SHOW_SUPER_STRUCTURE|SHOW_HYPER_STRUCTURE|GV_NODE_NAME_USES_GLOBAL_ID|SHOW_ALL_IDS|SHOW_ALL_SUPERIDS|SHOW_ALL_HYPERIDS|SHOW_EXTRA_DATA,
  //                             this->BlockId,
  //                             this->SweepValues
  //                             );
  return resultStream.str();
} // DebugPrint()


// debug routine
template <typename FieldType>
void HierarchicalAugmenter<FieldType>::DebugSave(std::string filename)
{ // DebugSave
  std::ofstream outstream(filename);
  outstream << "Augmented Tree:" << std::endl;
  vtkm::worklet::contourtree_augmented::IdArrayType temp;
  vtkm::cont::Algorithm::Copy(vtkm::cont::make_ArrayHandleConstant<vtkm::Id>(
                                0, this->AugmentedTree->Supernodes.GetNumberOfValues()),
                              temp);
  std::string dumpVolumesString =
    vtkm::worklet::contourtree_distributed::HierarchicalContourTree<FieldType>::DumpVolumes(
      this->AugmentedTree->Supernodes,
      this->AugmentedTree->Superarcs,
      this->AugmentedTree->RegularNodeGlobalIds,
      0,
      temp,
      temp);
  outstream << dumpVolumesString;
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Global Regular IDs", this->GlobalRegularIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintValues("Data Values", this->DataValues, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Supernode IDs", this->SupernodeIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Superparents", this->Superparents, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Superparent Rounds", this->SuperparentRounds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "WhichRounds", this->WhichRounds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Global Regular IDs", this->OutData.GlobalRegularIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintValues(
    "Out Data Values", this->OutData.DataValues, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Supernode IDs", this->OutData.SupernodeIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Superparents", this->OutData.Superparents, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out Superparent Rounds", this->OutData.SuperparentRounds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Out WhichRounds", this->OutData.WhichRounds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "In Global Regular IDs", this->InData.GlobalRegularIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintValues(
    "In Data Values", this->InData.DataValues, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "In Supernode IDs", this->InData.SupernodeIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "In Superparents", this->InData.Superparents, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "In Superparent Rounds", this->InData.SuperparentRounds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "In WhichRounds", this->InData.WhichRounds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "First Attach / Rd", this->FirstAttachmentPointInRound, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "AttachmentIDs", this->AttachmentIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "New Supernode IDs", this->NewSupernodeIds, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Kept Supernodes", this->KeptSupernodes, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Supernode Sorter", this->SupernodeSorter, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Global Regular ID", this->GlobalRegularIdSet, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintValues(
    "Data Values", this->DataValueSet, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Superparents", this->SuperparentSet, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "SupernodeIDs", this->SupernodeIdSet, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "Supernode ID", this->SupernodeSorter, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Superparent",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->SuperparentSet),
    -1,
    outstream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Value",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->DataValueSet),
    -1,
    outstream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Global ID",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->GlobalRegularIdSet),
    -1,
    outstream);
  vtkm::worklet::contourtree_augmented::PrintArrayHandle(
    "Permuted Supernode ID",
    vtkm::cont::make_ArrayHandlePermutation(this->SupernodeSorter, this->SupernodeIdSet),
    -1,
    outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "RegularNodesNeeded", this->RegularNodesNeeded, -1, outstream);
  vtkm::worklet::contourtree_augmented::PrintIndices(
    "RegularSuperparents", this->RegularSuperparents, -1, outstream);
} // DebugSave



} // namespace contourtree_distributed
} // namespace worklet
} // namespace vtkm


#endif
