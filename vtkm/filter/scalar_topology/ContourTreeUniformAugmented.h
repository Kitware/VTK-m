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

#ifndef vtk_m_filter_scalar_topology_ContourTreeUniformAugmented_h
#define vtk_m_filter_scalar_topology_ContourTreeUniformAugmented_h

#include <vtkm/Types.h>
#include <vtkm/cont/ArrayHandle.h>

#include <vtkm/filter/Filter.h>
#include <vtkm/filter/scalar_topology/vtkm_filter_scalar_topology_export.h>

#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/ContourTree.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/MultiBlockContourTreeHelper.h>

#include <memory>

namespace vtkm
{
namespace filter
{
namespace scalar_topology
{
/// \brief Construct the Contour Tree for a 2D or 3D regular mesh
///
/// This filter implements the parallel peak pruning algorithm. In contrast to
/// the ContourTreeUniform filter, this filter is optimized to allow for the
/// computation of the augmented contour tree, i.e., the contour tree including
/// all regular mesh vertices. Augmentation with regular vertices is used in
/// practice to compute statistics (e.g., volume), to segment the input mesh,
/// facilitate iso-value selection, enable localization of all verticies of a
/// mesh in the tree among others.
///
/// In addition to single-block computation, the filter also supports multi-block
/// regular grids. The blocks are processed in parallel using DIY and then the
/// tree are merged progressively using a binary-reduction scheme to compute the
/// final contour tree. I.e., in the multi-block context, the final tree is
/// constructed on rank 0.
class VTKM_FILTER_SCALAR_TOPOLOGY_EXPORT ContourTreeAugmented : public vtkm::filter::Filter
{
public:
  VTKM_CONT bool CanThread() const override
  {
    // shared helper object MultiBlockTreeHelper.
    // TODO: need further investigation.
    return false;
  }

  ///
  /// Create the contour tree filter
  /// @param[in] useMarchingCubes Boolean indicating whether marching cubes (true) or freudenthal (false)
  ///                             connectivity should be used. Valid only for 3D input data. Default is false.
  /// @param[in] computeRegularStructure  Unsigned int indicating whether the tree should be augmented.
  ///                             0=no augmentation, 1=full augmentation, 2=boundary augmentation. The
  ///                             latter option (=2) is mainly relevant for multi-block input data to
  ///                             improve efficiency by considering only boundary vertices during the
  ///                             merging of data blocks.
  ///
  VTKM_CONT
  explicit ContourTreeAugmented(bool useMarchingCubes = false,
                                unsigned int computeRegularStructure = 1);

  ///
  /// Define the spatial decomposition of the data in case we run in parallel with a multi-block dataset
  ///
  /// Note: Only used when running on a multi-block dataset.
  /// @param[in] blocksPerDim  Number of data blocks used in each data dimension
  /// @param[in] localBlockIndices  Array with the (x,y,z) index of each local data block with
  ///                               with respect to blocksPerDim
  VTKM_CONT
  void SetBlockIndices(vtkm::Id3 blocksPerDim,
                       const vtkm::cont::ArrayHandle<vtkm::Id3>& localBlockIndices);

  ///@{
  /// Get the contour tree computed by the filter
  const vtkm::worklet::contourtree_augmented::ContourTree& GetContourTree() const;
  /// Get the sort order for the mesh vertices
  const vtkm::worklet::contourtree_augmented::IdArrayType& GetSortOrder() const;
  /// Get the number of iterations used to compute the contour tree
  vtkm::Id GetNumIterations() const;
  ///@}

private:
  /// Output field "saddlePeak" wich is pairs of vertex ids indicating saddle and peak of contour
  VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input) override;
  VTKM_CONT vtkm::cont::PartitionedDataSet DoExecutePartitions(
    const vtkm::cont::PartitionedDataSet& inData) override;

  ///@{
  /// when operating on vtkm::cont::MultiBlock we want to
  /// do processing across ranks as well. Just adding pre/post handles
  /// for the same does the trick.
  VTKM_CONT void PreExecute(const vtkm::cont::PartitionedDataSet& input);

  VTKM_CONT void PostExecute(const vtkm::cont::PartitionedDataSet& input,
                             vtkm::cont::PartitionedDataSet& output);
  ///
  /// Internal helper function that implements the actual functionality of PostExecute
  ///
  /// In the case we operate on vtkm::cont::MultiBlock we need to merge the trees
  /// computed on the block to compute the final contour tree.
  template <typename T>
  VTKM_CONT void DoPostExecute(const vtkm::cont::PartitionedDataSet& input,
                               vtkm::cont::PartitionedDataSet& output);
  ///@}

  /// Use marching cubes connectivity for computing the contour tree
  bool UseMarchingCubes;
  // 0=no augmentation, 1=full augmentation, 2=boundary augmentation
  unsigned int ComputeRegularStructure;

  // TODO Should the additional fields below be add to the vtkm::filter::ResultField and what is the best way to represent them
  // Additional result fields not included in the vtkm::filter::ResultField returned by DoExecute

  /// The contour tree computed by the filter
  vtkm::worklet::contourtree_augmented::ContourTree ContourTreeData;
  /// Number of iterations used to compute the contour tree
  vtkm::Id NumIterations = 0;
  /// Array with the sorted order of the mesh vertices
  vtkm::worklet::contourtree_augmented::IdArrayType MeshSortOrder;
  /// Helper object to help with the parallel merge when running with DIY in parallel with MulitBlock data
  std::unique_ptr<vtkm::worklet::contourtree_distributed::MultiBlockContourTreeHelper>
    MultiBlockTreeHelper;
};
} // namespace scalar_topology
} // namespace filter
} // namespace vtkm

#endif // vtk_m_filter_scalar_topology_ContourTreeUniformAugmented_h
