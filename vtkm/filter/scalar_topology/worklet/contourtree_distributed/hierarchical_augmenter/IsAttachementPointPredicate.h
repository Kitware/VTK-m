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

#ifndef vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_is_attachement_point_predicate_h
#define vtk_m_worklet_contourtree_distributed_hierarchical_augmenter_is_attachement_point_predicate_h

#include <vtkm/Types.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ExecutionObjectBase.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/Types.h>

namespace vtkm
{
namespace worklet
{
namespace contourtree_distributed
{
namespace hierarchical_augmenter
{


/// Predicate used in HierarchicalAugmenter<FieldType>::Initalize to determine
/// whether a node is an attachement point
class IsAttachementPointPredicateImpl
{
public:
  using IdPortalType = vtkm::worklet::contourtree_augmented::IdArrayType::ReadPortalType;

  // constructor - takes vectors as parameters
  VTKM_CONT
  IsAttachementPointPredicateImpl(
    const vtkm::worklet::contourtree_augmented::IdArrayType& superarcs,
    const vtkm::worklet::contourtree_augmented::IdArrayType& whichRound,
    const vtkm::Id numRounds,
    vtkm::worklet::contourtree_augmented::IdArrayType* volumeArray,
    vtkm::Id presimplifyThreshold,
    vtkm::cont::DeviceAdapterId device,
    vtkm::cont::Token& token)
    : SuperarcsPortal(superarcs.PrepareForInput(device, token))
    , WhichRoundPortal(whichRound.PrepareForInput(device, token))
    , NumRounds(numRounds)
    , PresimplifyThreshold(presimplifyThreshold)
  { // constructor
    this->Presimplify = ((volumeArray != NULL) && (presimplifyThreshold > 0));
    // If we presimplify then store the volumeArray. Otherwise we don't need to volume array and we
    // set it to another portal, just to make sure the variable is being initalized with something
    this->VolumeArrayPortal =
      this->Presimplify ? volumeArray->PrepareForInput(device, token) : this->WhichRoundPortal;
  } // constructor

  // () operator - gets called to do comparison
  VTKM_EXEC
  bool operator()(const vtkm::Id& supernode) const
  { // operator()
    // an attachment point is defined by having no superarc (NO_SUCH_ELEMENT) and not being in
    // the final round (where this indicates the global root)
    bool predicate =
      (vtkm::worklet::contourtree_augmented::NoSuchElement(this->SuperarcsPortal.Get(supernode)) &&
       (this->WhichRoundPortal.Get(supernode) < this->NumRounds));
    // if we pass this check then we need to also check that the supernode passes the pre-simplification threshold
    if (predicate && this->Presimplify)
    {
      // suppress if it's volume is at or below the threshold
      if (this->VolumeArrayPortal.Get(supernode) <= this->PresimplifyThreshold)
      {                    // below threshold
        predicate = false; // do not keep attachement point below the simplification threshold
      }                    // below threshold
    }
    return predicate;
  } // operator()

private:
  IdPortalType SuperarcsPortal;
  IdPortalType WhichRoundPortal;
  const vtkm::Id NumRounds;
  bool Presimplify;
  IdPortalType VolumeArrayPortal;
  vtkm::Id PresimplifyThreshold;


}; // IsAttachementPointPredicateImpl

class IsAttachementPointPredicate : public vtkm::cont::ExecutionObjectBase
{
public:
  // constructor - takes vectors as parameters
  VTKM_CONT
  IsAttachementPointPredicate(const vtkm::worklet::contourtree_augmented::IdArrayType& superarcs,
                              const vtkm::worklet::contourtree_augmented::IdArrayType& whichRound,
                              const vtkm::Id numRounds,
                              vtkm::worklet::contourtree_augmented::IdArrayType* volumeArray = NULL,
                              vtkm::Id presimplifyThreshold = 0)
    : Superarcs(superarcs)
    , WhichRound(whichRound)
    , NumRounds(numRounds)
    , VolumeArray(volumeArray)
    , PresimplifyThreshold(presimplifyThreshold)
  {
  }

  VTKM_CONT IsAttachementPointPredicateImpl PrepareForExecution(vtkm::cont::DeviceAdapterId device,
                                                                vtkm::cont::Token& token) const
  {
    return IsAttachementPointPredicateImpl(this->Superarcs,
                                           this->WhichRound,
                                           this->NumRounds,
                                           this->VolumeArray,
                                           this->PresimplifyThreshold,
                                           device,
                                           token);
  }

private:
  vtkm::worklet::contourtree_augmented::IdArrayType Superarcs;
  vtkm::worklet::contourtree_augmented::IdArrayType WhichRound;
  const vtkm::Id NumRounds;
  vtkm::worklet::contourtree_augmented::IdArrayType* VolumeArray;
  vtkm::Id PresimplifyThreshold;
}; // IsAttachementPointPredicate

} // namespace hierarchical_augmenter
} // namespace contourtree_augmented
} // namespace worklet
} // namespace vtkm

#endif
