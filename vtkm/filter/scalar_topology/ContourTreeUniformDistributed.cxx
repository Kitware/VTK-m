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

// vtkm includes
#include <vtkm/cont/CellSetStructured.h>
#include <vtkm/cont/ErrorFilterExecution.h>
#include <vtkm/cont/Timer.h>

// single-node augmented contour tree includes
#include <vtkm/filter/scalar_topology/ContourTreeUniformDistributed.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/DataSetMesh.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/meshtypes/ContourTreeMesh.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_augmented/meshtypes/mesh_boundary/MeshBoundaryContourTreeMesh.h>

// distributed contour tree includes
#include <vtkm/filter/scalar_topology/internal/ComputeBlockIndices.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/BoundaryTree.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/BoundaryTreeMaker.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/CombineHyperSweepBlockFunctor.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/ComputeDistributedContourTreeFunctor.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/DistributedContourTreeBlockData.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/HierarchicalAugmenterFunctor.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/HierarchicalHyperSweeper.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/HyperSweepBlock.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/InteriorForest.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/PrintGraph.h>
#include <vtkm/filter/scalar_topology/worklet/contourtree_distributed/TreeGrafter.h>

// DIY includes
// clang-format off
VTKM_THIRDPARTY_PRE_INCLUDE
#include <vtkm/thirdparty/diy/Configure.h>
#include <vtkm/thirdparty/diy/diy.h>
VTKM_THIRDPARTY_POST_INCLUDE
// clang-format on

namespace vtkm
{
namespace filter
{
namespace scalar_topology
{
//-----------------------------------------------------------------------------
// Helper structs needed to support approbriate type discovery as part
// of pre- and post-execute
//-----------------------------------------------------------------------------
namespace contourtree_distributed_detail
{
/// Helper function for saving the content of the tree for debugging
template <typename FieldType>
void SaveAfterFanInResults(
  vtkm::worklet::contourtree_distributed::DistributedContourTreeBlockData<FieldType>* blockData,
  vtkm::Id rank,
  vtkm::cont::LogLevel logLevel)
{
  (void)logLevel; // Suppress unused variable warning if logging is disabled
  VTKM_LOG_S(logLevel,
             "Fan In Complete" << std::endl
                               << "# of CTs: " << blockData->ContourTrees.size() << std::endl
                               << "# of CTMs: " << blockData->ContourTreeMeshes.size() << std::endl
                               << "# of IFs: " << blockData->InteriorForests.size() << std::endl);

  char buffer[256];
  std::snprintf(buffer,
                sizeof(buffer),
                "AfterFanInResults_Rank%d_Block%d.txt",
                static_cast<int>(rank),
                static_cast<int>(blockData->LocalBlockNo));
  std::ofstream os(buffer);
  os << "Contour Trees" << std::endl;
  os << "=============" << std::endl;
  for (const auto& ct : blockData->ContourTrees)
    ct.PrintContent(os);
  os << std::endl;
  os << "Contour Tree Meshes" << std::endl;
  os << "===================" << std::endl;
  for (const auto& cm : blockData->ContourTreeMeshes)
    cm.PrintContent(os);
  os << std::endl;
  os << "Interior Forests" << std::endl;
  os << "===================" << std::endl;
  for (const auto& info : blockData->InteriorForests)
    info.PrintContent(os);
  os << std::endl;
}

template <typename FieldType>
void SaveHierarchicalTreeDot(
  const vtkm::worklet::contourtree_distributed::DistributedContourTreeBlockData<FieldType>*
    blockData,
  vtkm::Id rank,
  vtkm::Id nRounds)
{
  std::string hierarchicalTreeFileName = std::string("Rank_") +
    std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
    std::to_string(static_cast<int>(blockData->LocalBlockNo)) + std::string("_Round_") +
    std::to_string(nRounds) + std::string("_Hierarchical_Tree.gv");
  std::string hierarchicalTreeLabel = std::string("Block ") +
    std::to_string(static_cast<int>(blockData->LocalBlockNo)) + std::string(" Round ") +
    std::to_string(nRounds) + std::string(" Hierarchical Tree");
  vtkm::Id hierarchicalTreeDotSettings =
    vtkm::worklet::contourtree_distributed::SHOW_SUPER_STRUCTURE |
    vtkm::worklet::contourtree_distributed::SHOW_HYPER_STRUCTURE |
    vtkm::worklet::contourtree_distributed::SHOW_ALL_IDS |
    vtkm::worklet::contourtree_distributed::SHOW_ALL_SUPERIDS |
    vtkm::worklet::contourtree_distributed::SHOW_ALL_HYPERIDS;
  std::ofstream hierarchicalTreeFile(hierarchicalTreeFileName);
  hierarchicalTreeFile
    << vtkm::worklet::contourtree_distributed::HierarchicalContourTreeDotGraphPrint<FieldType>(
         hierarchicalTreeLabel, blockData->HierarchicalTree, hierarchicalTreeDotSettings);
}

} // end namespace contourtree_distributed_detail

//-----------------------------------------------------------------------------
// Main constructor
//-----------------------------------------------------------------------------
ContourTreeUniformDistributed::ContourTreeUniformDistributed(vtkm::cont::LogLevel timingsLogLevel,
                                                             vtkm::cont::LogLevel treeLogLevel)
  : UseBoundaryExtremaOnly(true)
  , UseMarchingCubes(false)
  , AugmentHierarchicalTree(false)
  , PresimplifyThreshold(0)
  , SaveDotFiles(false)
  , TimingsLogLevel(timingsLogLevel)
  , TreeLogLevel(treeLogLevel)
  , BlocksPerDimension(vtkm::Id3{ -1, -1, -1 })
  , LocalBlockIndices()
  , LocalMeshes()
  , LocalContourTrees()
  , LocalBoundaryTrees()
  , LocalInteriorForests()
{
  this->SetOutputFieldName("resultData");
}

//-----------------------------------------------------------------------------
// Functions used in PrepareForExecution() to compute the local contour
// tree for the data blocks processed by this rank.
//-----------------------------------------------------------------------------
template <typename T, typename StorageType>
void ContourTreeUniformDistributed::ComputeLocalTree(
  const vtkm::Id blockIndex,
  const vtkm::cont::DataSet& input,
  const vtkm::cont::ArrayHandle<T, StorageType>& fieldArray)
{
  // Get mesh size
  vtkm::Id3 meshSize;
  const auto& cells = input.GetCellSet();
  cells.CastAndCallForTypes<VTKM_DEFAULT_CELL_SET_LIST_STRUCTURED>(
    vtkm::worklet::contourtree_augmented::GetPointDimensions(), meshSize);

  // Create the mesh we need for the contour tree computation so that we have access to it
  // afterwards to compute the BRACT for each data block as well
  if (meshSize[2] == 1) // 2D mesh
  {
    vtkm::worklet::contourtree_augmented::DataSetMeshTriangulation2DFreudenthal mesh(
      vtkm::Id2{ meshSize[0], meshSize[1] });
    this->LocalMeshes[static_cast<std::size_t>(blockIndex)] = mesh;
    auto meshBoundaryExecObject = mesh.GetMeshBoundaryExecutionObject();
    this->ComputeLocalTreeImpl(blockIndex, input, fieldArray, mesh, meshBoundaryExecObject);
  }
  else if (this->UseMarchingCubes) // 3D marching cubes mesh
  {
    vtkm::worklet::contourtree_augmented::DataSetMeshTriangulation3DMarchingCubes mesh(meshSize);
    this->LocalMeshes[static_cast<std::size_t>(blockIndex)] = mesh;
    auto meshBoundaryExecObject = mesh.GetMeshBoundaryExecutionObject();
    this->ComputeLocalTreeImpl(blockIndex, input, fieldArray, mesh, meshBoundaryExecObject);
  }
  else // Regular 3D mesh
  {
    vtkm::worklet::contourtree_augmented::DataSetMeshTriangulation3DFreudenthal mesh(meshSize);
    this->LocalMeshes[static_cast<std::size_t>(blockIndex)] = mesh;
    auto meshBoundaryExecObject = mesh.GetMeshBoundaryExecutionObject();
    this->ComputeLocalTreeImpl(blockIndex, input, fieldArray, mesh, meshBoundaryExecObject);
  }
} // ContourTreeUniformDistributed::ComputeLocalTree

//-----------------------------------------------------------------------------
template <typename T, typename StorageType, typename MeshType, typename MeshBoundaryExecType>
void ContourTreeUniformDistributed::ComputeLocalTreeImpl(
  const vtkm::Id blockIndex,
  const vtkm::cont::DataSet& ds, // input,
  const vtkm::cont::ArrayHandle<T, StorageType>& field,
  MeshType& mesh,
  MeshBoundaryExecType& meshBoundaryExecObject)
{
  vtkm::cont::Timer timer;
  timer.Start();
  // We always need to compute the fully augmented contour tree for our local data block
  const unsigned int compRegularStruct = 1;

  // Set up the worklet
  vtkm::worklet::ContourTreeAugmented worklet;
  worklet.TimingsLogLevel = vtkm::cont::LogLevel::Off; // turn of the loggin, we do this afterwards
  worklet.Run(field,
              mesh,
              this->LocalContourTrees[static_cast<std::size_t>(blockIndex)],
              this->LocalMeshes[static_cast<std::size_t>(blockIndex)].SortOrder,
              this->NumIterations,
              compRegularStruct,
              meshBoundaryExecObject);
  // Log the contour tree timiing stats
  VTKM_LOG_S(this->TimingsLogLevel,
             std::endl
               << "    ---------------- Contour Tree Worklet Timings ------------------"
               << std::endl
               << "    Block Index : " << blockIndex << std::endl
               << worklet.TimingsLogString);
  VTKM_LOG_S(this->TimingsLogLevel,
             std::endl
               << "    "
                  "ComputeLocalTree ContourTree (blockIndex="
               << blockIndex << ") "
               << ": " << timer.GetElapsedTime() << " seconds");
  timer.Start();
  // Now we compute the BRACT for our data block. We do this here because we know the MeshType
  // here and we don't need to store the mesh separately any more since it is stored in the BRACT

  // Get the mesh information needed to create an IdRelabeler to relable local to global ids
  // Create an IdRelabeler since we are using a DataSetMesh type here, we don't need
  // the IdRelabeler for the BRACT construction when we are using a ContourTreeMesh.

  vtkm::Id3 pointDimensions, globalPointDimensions, globalPointIndexStart;
  ds.GetCellSet().CastAndCallForTypes<VTKM_DEFAULT_CELL_SET_LIST_STRUCTURED>(
    vtkm::worklet::contourtree_augmented::GetLocalAndGlobalPointDimensions(),
    pointDimensions,
    globalPointDimensions,
    globalPointIndexStart);
  auto localToGlobalIdRelabeler = vtkm::worklet::contourtree_augmented::mesh_dem::IdRelabeler(
    globalPointIndexStart, pointDimensions, globalPointDimensions);
  // Initialize the BoundaryTreeMaker
  auto boundaryTreeMaker =
    vtkm::worklet::contourtree_distributed::BoundaryTreeMaker<MeshType, MeshBoundaryExecType>(
      &mesh,                                                         // The input mesh
      meshBoundaryExecObject,                                        // The mesh boundary
      this->LocalContourTrees[static_cast<std::size_t>(blockIndex)], // The contour tree
      &this->LocalBoundaryTrees[static_cast<std::size_t>(
        blockIndex)], // The boundary tree (a.k.a BRACT) to be computed
      &this->LocalInteriorForests[static_cast<std::size_t>(
        blockIndex)] // The interior forest (a.k.a. Residue) to be computed
    );
  // Execute the BRACT construction, including the compute of the InteriorForest
  boundaryTreeMaker.Construct(&localToGlobalIdRelabeler, this->UseBoundaryExtremaOnly);
  // Log timing statistics
  VTKM_LOG_S(this->TimingsLogLevel,
             std::endl
               << "    "
                  "ComputeLocalTree BoundaryTreeMaker (blockIndex="
               << blockIndex << ") "
               << ": " << timer.GetElapsedTime() << " seconds");
  timer.Start();

  // At this point, I'm reasonably certain that the contour tree has been computed regardless of data push/pull
  // So although it might be logical to print things out earlier, I'll do it here
  // save the regular structure
  if (this->SaveDotFiles)
  {
    // Get the rank
    vtkm::Id rank = vtkm::cont::EnvironmentTracker::GetCommunicator().rank();

    // Save the BRACT dot for debug
    { // make context so the file will be closed and potentially large strings are cleaned up
      std::string bractFileName = std::string("Rank_") + std::to_string(static_cast<int>(rank)) +
        std::string("_Block_") + std::to_string(static_cast<int>(blockIndex)) + "_Initial_BRACT.gv";
      std::ofstream bractFile(bractFileName);
      std::string bractString =
        this->LocalBoundaryTrees[static_cast<std::size_t>(blockIndex)].PrintGlobalDot(
          "Before Fan In",
          mesh,
          field,
          globalPointIndexStart,
          pointDimensions,
          globalPointDimensions);
      bractFile << bractString << std::endl;
    }

    // Save the regular structure as a dot file
    { // make context so the file will be closed and potentially large strings are cleaned up
      std::string regularStructureFileName = std::string("Rank_") +
        std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
        std::to_string(static_cast<int>(blockIndex)) +
        std::string("_Initial_Step_0_Contour_Tree_Regular_Structure.gv");
      std::ofstream regularStructureFile(regularStructureFileName);
      std::string label = std::string("Block ") +
        std::to_string(static_cast<std::size_t>(blockIndex)) +
        " Initial Step 0 Contour Tree Regular Structure";
      vtkm::Id dotSettings = worklet::contourtree_distributed::SHOW_REGULAR_STRUCTURE |
        worklet::contourtree_distributed::SHOW_ALL_IDS;
      std::string regularStructureString =
        worklet::contourtree_distributed::ContourTreeDotGraphPrint<
          T,
          StorageType,
          MeshType,
          vtkm::worklet::contourtree_augmented::IdArrayType>(
          label, // graph title
          static_cast<MeshType&>(this->LocalMeshes[static_cast<std::size_t>(
            blockIndex)]),           // the underlying mesh for the contour tree
          &localToGlobalIdRelabeler, // relabler needed to compute global ids
          field,                     // data values
          this->LocalContourTrees[static_cast<std::size_t>(blockIndex)], // local contour tree
          dotSettings // mask with flags for what elements to show
        );
      regularStructureFile << regularStructureString << std::endl;
    }

    // Save the super structure as a dot file
    { // make context so the file will be closed and potentially large strings are cleaned up
      std::string superStructureFileName = std::string("Rank_") +
        std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
        std::to_string(static_cast<int>(blockIndex)) +
        std::string("_Initial_Step_1_Contour_Tree_Super_Structure.gv");
      std::ofstream superStructureFile(superStructureFileName);
      vtkm::Id ctPrintSettings = worklet::contourtree_distributed::SHOW_SUPER_STRUCTURE |
        worklet::contourtree_distributed::SHOW_HYPER_STRUCTURE |
        worklet::contourtree_distributed::SHOW_ALL_IDS |
        worklet::contourtree_distributed::SHOW_ALL_SUPERIDS |
        worklet::contourtree_distributed::SHOW_ALL_HYPERIDS;
      std::string ctPrintLabel = std::string("Block ") +
        std::to_string(static_cast<size_t>(blockIndex)) +
        " Initial Step 1 Contour Tree Super Structure";
      std::string superStructureString =
        vtkm::worklet::contourtree_distributed::ContourTreeDotGraphPrint<
          T,
          StorageType,
          MeshType,
          vtkm::worklet::contourtree_augmented::IdArrayType>(
          ctPrintLabel,
          static_cast<MeshType&>(this->LocalMeshes[static_cast<std::size_t>(blockIndex)]),
          &localToGlobalIdRelabeler,
          field,
          this->LocalContourTrees[static_cast<std::size_t>(blockIndex)],
          ctPrintSettings);
      superStructureFile << superStructureString << std::endl;
    }

    // save the Boundary Tree as a dot file
    { // make context so the file will be closed and potentially large strings are cleaned up
      std::string boundaryTreeFileName = std::string("Rank_") +
        std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
        std::to_string(static_cast<size_t>(blockIndex)) +
        std::string("_Initial_Step_3_Boundary_Tree.gv");
      std::ofstream boundaryTreeFile(boundaryTreeFileName);
      std::string boundaryTreeString =
        vtkm::worklet::contourtree_distributed::BoundaryTreeDotGraphPrint(
          std::string("Block ") + std::to_string(static_cast<size_t>(blockIndex)) +
            std::string(" Initial Step 3 Boundary Tree"),
          static_cast<MeshType&>(this->LocalMeshes[static_cast<std::size_t>(blockIndex)]),
          meshBoundaryExecObject,
          this->LocalBoundaryTrees[static_cast<std::size_t>(blockIndex)],
          &localToGlobalIdRelabeler,
          field);
      boundaryTreeFile << boundaryTreeString << std::endl;
    }

    // and save the Interior Forest as another dot file
    { // make context so the file will be closed and potentially large strings are cleaned up
      std::string interiorForestFileName = std::string("Rank_") +
        std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
        std::to_string(static_cast<int>(blockIndex)) +
        std::string("_Initial_Step_4_Interior_Forest.gv");
      std::ofstream interiorForestFile(interiorForestFileName);
      std::string interiorForestString =
        worklet::contourtree_distributed::InteriorForestDotGraphPrint(
          std::string("Block ") + std::to_string(rank) + " Initial Step 4 Interior Forest",
          this->LocalInteriorForests[static_cast<std::size_t>(blockIndex)],
          this->LocalContourTrees[static_cast<std::size_t>(blockIndex)],
          this->LocalBoundaryTrees[static_cast<std::size_t>(blockIndex)],
          static_cast<MeshType&>(this->LocalMeshes[static_cast<std::size_t>(blockIndex)]),
          meshBoundaryExecObject,
          &localToGlobalIdRelabeler,
          field);
      interiorForestFile << interiorForestString << std::endl;

      // Log timing statistics
      VTKM_LOG_S(this->TimingsLogLevel,
                 std::endl
                   << "    " << std::setw(38) << std::left << "ComputeLocalTree Save Dot"
                   << ": " << timer.GetElapsedTime() << " seconds");
    }
  } // if (this->SaveDotFiles)
} // ContourTreeUniformDistributed::ComputeLocalTreeImpl


vtkm::cont::DataSet ContourTreeUniformDistributed::DoExecute(const vtkm::cont::DataSet& input)
{
  vtkm::cont::PartitionedDataSet output = this->Execute(vtkm::cont::PartitionedDataSet(input));
  if (output.GetNumberOfPartitions() > 1)
  {
    throw vtkm::cont::ErrorFilterExecution("Expecting at most 1 block.");
  }
  return output.GetNumberOfPartitions() == 1 ? output.GetPartition(0) : vtkm::cont::DataSet();
}

//-----------------------------------------------------------------------------
// Main execution phases of the filter
//
// The functions are executed by VTKm in the following order
// - PreExecute
// - PrepareForExecution
//   --> ComputeLocalTree (struct)
//      --> ComputeLocalTree (filter funct)
//        --> ComputeLocalTreeImpl (filter funct)
// - PostExecute
//   --> DoPostExecute
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
VTKM_CONT void ContourTreeUniformDistributed::PreExecute(
  const vtkm::cont::PartitionedDataSet& input)
{
  // TODO/FIXME: The following may be too expensive for a "sanity" check as it
  // requires global communication
  auto globalNumberOfPartitions = input.GetGlobalNumberOfPartitions();

  if (globalNumberOfPartitions < 2)
  {
    throw vtkm::cont::ErrorFilterExecution("ContourTreeUniformDistributed filter expects a "
                                           "PartitionedDataSet with at least two partitions.");
  }

  if (this->BlocksPerDimension[0] != -1)
  {
    if (this->BlocksPerDimension[1] < 1 || this->BlocksPerDimension[2] < 1)
    {
      throw vtkm::cont::ErrorFilterExecution("Invalid input BlocksPerDimension.");
    }
    if (globalNumberOfPartitions !=
        this->BlocksPerDimension[0] * this->BlocksPerDimension[1] * this->BlocksPerDimension[2])
    {
      throw vtkm::cont::ErrorFilterExecution("Global number of blocks in data set does not match "
                                             "expected value based on BlocksPerDimension");
    }
    if (this->LocalBlockIndices.GetNumberOfValues() != input.GetNumberOfPartitions())
    {
      throw vtkm::cont::ErrorFilterExecution("Local number of partitions in data set does not "
                                             "match number of specified blocks indices.");
    }
  }

  // Allocate vectors
  this->LocalMeshes.resize(static_cast<std::size_t>(input.GetGlobalNumberOfPartitions()));
  this->LocalContourTrees.resize(static_cast<std::size_t>(input.GetGlobalNumberOfPartitions()));
  this->LocalBoundaryTrees.resize(static_cast<std::size_t>(input.GetGlobalNumberOfPartitions()));
  this->LocalInteriorForests.resize(static_cast<std::size_t>(input.GetGlobalNumberOfPartitions()));
}

vtkm::cont::PartitionedDataSet ContourTreeUniformDistributed::DoExecutePartitions(
  const vtkm::cont::PartitionedDataSet& input)
{
  // Time execution
  vtkm::cont::Timer timer;
  timer.Start();

  this->PreExecute(input);

  // Compute the local contour tree, boundary tree, and interior forest for each local data block
  for (vtkm::Id blockNo = 0; blockNo < input.GetNumberOfPartitions(); ++blockNo)
  {
    const auto& dataset = input.GetPartition(blockNo);
    const auto& field =
      dataset.GetField(this->GetActiveFieldName(), this->GetActiveFieldAssociation());
    if (!field.IsPointField())
    {
      throw vtkm::cont::ErrorFilterExecution("Point field expected.");
    }

    this->CastAndCallScalarField(
      field, [&](const auto concrete) { this->ComputeLocalTree(blockNo, dataset, concrete); });
  }

  // Log sizes of the local contour trees, boundary trees, and interior forests
  for (size_t bi = 0; bi < this->LocalContourTrees.size(); bi++)
  {
    VTKM_LOG_S(this->TreeLogLevel,
               std::endl
                 << "    ---------------- Contour Tree Array Sizes ---------------------"
                 << std::endl
                 << "    Block Index : " << bi << std::endl
                 << LocalContourTrees[bi].PrintArraySizes());
    VTKM_LOG_S(this->TreeLogLevel,
               std::endl
                 << "    ---------------- Boundary Tree Array Sizes ---------------------"
                 << std::endl
                 << "    Block Index : " << bi << std::endl
                 << LocalBoundaryTrees[bi].PrintArraySizes());
    VTKM_LOG_S(this->TreeLogLevel,
               std::endl
                 << "    ---------------- Interior Forest Array Sizes ---------------------"
                 << std::endl
                 << "    Block Index : " << bi << std::endl
                 << LocalInteriorForests[bi].PrintArraySizes());
    // VTKM_LOG_S(this->TreeLogLevel,
    //           std::endl
    //           << "    ---------------- Hyperstructure Statistics ---------------------"  << std::endl
    //           << LocalContourTrees[bi].PrintHyperStructureStatistics(false) << std::endl);
  }

  // Log timing statistics
  VTKM_LOG_S(this->TimingsLogLevel,
             std::endl
               << "    " << std::setw(38) << std::left << "Contour Tree Filter PrepareForExecution"
               << ": " << timer.GetElapsedTime() << " seconds");

  vtkm::cont::PartitionedDataSet result;
  this->PostExecute(input, result);

  return result;
}

//-----------------------------------------------------------------------------
VTKM_CONT void ContourTreeUniformDistributed::PostExecute(
  const vtkm::cont::PartitionedDataSet& input,
  vtkm::cont::PartitionedDataSet& result)
{
  vtkm::cont::Timer timer;
  timer.Start();

  auto field = // TODO/FIXME: Correct for more than one block per rank?
    input.GetPartition(0).GetField(this->GetActiveFieldName(), this->GetActiveFieldAssociation());

  auto PostExecuteCaller = [&](const auto& concrete) {
    using T = typename std::decay_t<decltype(concrete)>::ValueType;
    this->DoPostExecute<T>(input, result);
  };
  this->CastAndCallScalarField(field, PostExecuteCaller);

  VTKM_LOG_S(this->TimingsLogLevel,
             std::endl
               << "    " << std::setw(38) << std::left << "Contour Tree Filter PostExecute"
               << ": " << timer.GetElapsedTime() << " seconds");
}

template <typename FieldType>
inline VTKM_CONT void ContourTreeUniformDistributed::ComputeVolumeMetric(
  vtkmdiy::Master& inputContourTreeMaster,
  vtkmdiy::DynamicAssigner& assigner,
  vtkmdiy::RegularSwapPartners& partners,
  const FieldType&, // dummy parameter to get the type
  std::stringstream& timingsStream,
  const vtkm::cont::PartitionedDataSet& input,
  bool useAugmentedTree,
  std::vector<vtkm::cont::ArrayHandle<vtkm::Id>>& intrinsicVolumes,
  std::vector<vtkm::cont::ArrayHandle<vtkm::Id>>& dependentVolumes)
{
  // TODO/FIXME: CONSIDER MOVING CONTENTS OF THIS METHOD TO SEPARATE FILTER
  vtkm::cont::Timer timer;
  timer.Start();

  using HyperSweepBlock = vtkm::worklet::contourtree_distributed::HyperSweepBlock<FieldType>;
  auto comm = vtkm::cont::EnvironmentTracker::GetCommunicator();
  vtkmdiy::Master hierarchical_hyper_sweep_master(comm,
                                                  1,  // Use 1 thread, VTK-M will do the treading
                                                  -1, // All blocks in memory
                                                  0,  // No create function
                                                  HyperSweepBlock::Destroy);

  // Log the time to create the DIY master for the hyper sweep
  timingsStream << "    " << std::setw(38) << std::left << "Create DIY Master (Hypersweep)"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // Copy data from hierarchical tree computation to initialize volume computation
  using DistributedContourTreeBlockData =
    vtkm::worklet::contourtree_distributed::DistributedContourTreeBlockData<FieldType>;
  inputContourTreeMaster.foreach (
    [&](DistributedContourTreeBlockData* currInBlock, const vtkmdiy::Master::ProxyWithLink&) {
      vtkm::Id blockNo = currInBlock->LocalBlockNo;
      //const vtkm::cont::DataSet& currDS = hierarchicalTreeOutputDataSet[blockNo];
      auto currOriginalBlock = input.GetPartition(static_cast<vtkm::Id>(blockNo));
      // The block size and origin may be modified during the FanIn so we need to use the
      // size and origin from the original decomposition instead of looking it up in the currInBlock
      vtkm::Id3 pointDimensions, globalPointDimensions, globalPointIndexStart;
      currOriginalBlock.GetCellSet().CastAndCallForTypes<VTKM_DEFAULT_CELL_SET_LIST_STRUCTURED>(
        vtkm::worklet::contourtree_augmented::GetLocalAndGlobalPointDimensions(),
        pointDimensions,
        globalPointDimensions,
        globalPointIndexStart);

      // NOTE: Use dummy link to make DIY happy. The dummy link is never used, since all
      //       communication is via RegularDecomposer, which sets up its own links. No need
      //       to keep the pointer, as DIY will "own" it and delete it when no longer needed.
      // NOTE: Since we passed a "Destroy" function to DIY master, it will own the local data
      //       blocks and delete them when done.

      // If we are pre-simplifying the tree then we need to use the base tree and if we compute the
      // final volume, then we need to use the augmented tree
      // currInBlock->HierarchicalAugmenter is NOT initialized
      //     when this function is first called if pre-simplification is applied.
      // currInBlock->HierarchicalAugmenter.AugmentedTree seems ok to remain,
      //     because it is only called during augmentation,
      //     in which the HierarchicalAugmenter is intialized.
      auto hierarchicalTreeToProcess = useAugmentedTree
        ? currInBlock->HierarchicalAugmenter.AugmentedTree
        : &currInBlock->HierarchicalTree;

#ifdef DEBUG_PRINT_HYPER_SWEEPER
      {
        std::stringstream debugStream;
        debugStream << "Block " << blockNo << std::endl;
        debugStream << hierarchicalTreeToProcess->DebugPrint(
          "Choosing Hierarchical Tree To Process", __FILE__, __LINE__);
        VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream);
      }
#endif // DEBUG_PRINT_HYPER_SWEEPER


      // Create HyperSweeper
      hierarchical_hyper_sweep_master.add(currInBlock->GlobalBlockId,
                                          new HyperSweepBlock(blockNo,
                                                              currInBlock->GlobalBlockId,
                                                              globalPointIndexStart,
                                                              pointDimensions,
                                                              globalPointDimensions,
                                                              *hierarchicalTreeToProcess),
                                          new vtkmdiy::Link());
    });

  // Log time to copy the data to the HyperSweepBlock data objects
  timingsStream << "    " << std::setw(38) << std::left << "Initialize Hypersweep Data"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  vtkmdiy::fix_links(hierarchical_hyper_sweep_master, assigner);

  // Record time to fix the links
  timingsStream << "    " << std::setw(38) << std::left << "Fix DIY Links (Hypersweep)"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  hierarchical_hyper_sweep_master.foreach (
    [&](HyperSweepBlock* b, const vtkmdiy::Master::ProxyWithLink&) {
      std::stringstream localHypersweepTimingsStream;
      vtkm::cont::Timer localHypersweepTimer;
      localHypersweepTimer.Start();

    // Create HyperSweeper
#ifdef DEBUG_PRINT_HYPER_SWEEPER
      {
        std::stringstream debugStream;
        const vtkm::Id nBlockVertices =
          b->Size[0] * b->Size[1] * (b->GlobalSize[2] <= 1 ? 1 : b->Size[2]);
        debugStream << std::endl;
        debugStream << std::endl;
        debugStream << std::endl;
        debugStream << std::endl;
        debugStream << "------------------------------" << std::endl;
        debugStream << "Computing Contour Tree Volumes" << std::endl;
        debugStream << "------------------------------" << std::endl;
        debugStream << std::endl;
        debugStream << std::endl;
        debugStream << "Volumes Before Initialisation" << std::endl;
        debugStream << "Block: " << b->GlobalBlockId << " Size: " << nBlockVertices << std::endl;
        vtkm::worklet::contourtree_augmented::PrintHeader(b->IntrinsicVolume.GetNumberOfValues(),
                                                          debugStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Intrinsic", b->IntrinsicVolume, -1, debugStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Dependent", b->DependentVolume, -1, debugStream);
        VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
      }
#endif

      // Create the HierarchicalHypersweeper
      vtkm::worklet::contourtree_distributed::HierarchicalHyperSweeper<vtkm::Id, FieldType>
        hyperSweeper(
          b->GlobalBlockId, b->HierarchicalContourTree, b->IntrinsicVolume, b->DependentVolume);
      // Log the time
      localHypersweepTimingsStream << "    Create Hypersweeper (block=" << b->LocalBlockNo
                                   << ") : " << localHypersweepTimer.GetElapsedTime() << " seconds"
                                   << std::endl;
      localHypersweepTimer.Start();

      // Create mesh and initialize vertex counts
      vtkm::worklet::contourtree_augmented::mesh_dem::IdRelabeler idRelabeler{ b->Origin,
                                                                               b->Size,
                                                                               b->GlobalSize };

#ifdef DEBUG_PRINT_HYPER_SWEEPER
      {
        std::stringstream debugStream;
        debugStream << "Computing Intrinsic Vertex Count" << std::endl;
        debugStream << "Block: " << b->GlobalBlockId << " Size: " << nBlockVertices << std::endl;
        VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
      }
#endif

      if (b->GlobalSize[2] <= 1)
      {
        vtkm::worklet::contourtree_augmented::DataSetMeshTriangulation2DFreudenthal mesh(
          vtkm::Id2{ b->Size[0], b->Size[1] });
        hyperSweeper.InitializeIntrinsicVertexCount(
          b->HierarchicalContourTree, mesh, idRelabeler, b->IntrinsicVolume);
      }
      else
      {
        // For getting owned vertices, it does not make a difference if we are using marching cubes or not.
        vtkm::worklet::contourtree_augmented::DataSetMeshTriangulation3DFreudenthal mesh(b->Size);
        hyperSweeper.InitializeIntrinsicVertexCount(
          b->HierarchicalContourTree, mesh, idRelabeler, b->IntrinsicVolume);
      }

      // Initialize dependentVolume by copy from intrinsicVolume
      vtkm::cont::Algorithm::Copy(b->IntrinsicVolume, b->DependentVolume);

#ifdef DEBUG_PRINT_HYPER_SWEEPER
      {
        std::stringstream debugStream;
        debugStream << "Intrinsic Volume Computed & Copied to Dependent" << std::endl;
        vtkm::worklet::contourtree_augmented::PrintHeader(b->IntrinsicVolume.GetNumberOfValues(),
                                                          debugStream);
        vtkm::cont::ArrayHandle<vtkm::Id> whichTreeSupernodeRegularIDs;
        // we copy the HCT information to a temp array because b->HierarchicalContourTree is a const
        vtkm::cont::ArrayHandle<vtkm::Id> hctGRIds;
        vtkm::cont::ArrayHandle<vtkm::Id> hctSupernodes;
        vtkm::cont::Algorithm::Copy(b->HierarchicalContourTree.RegularNodeGlobalIds, hctGRIds);
        vtkm::cont::Algorithm::Copy(b->HierarchicalContourTree.Supernodes, hctSupernodes);
        vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id>(
          hctGRIds, hctSupernodes, whichTreeSupernodeRegularIDs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Regular ID", whichTreeSupernodeRegularIDs, -1, debugStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Intrinsic", b->IntrinsicVolume, -1, debugStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Dependent", b->DependentVolume, -1, debugStream);
        VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
      }
#endif

      // Log the time
      localHypersweepTimingsStream << "    Initalize Vertex Counts (block=" << b->LocalBlockNo
                                   << ") : " << localHypersweepTimer.GetElapsedTime() << " seconds"
                                   << std::endl;
      localHypersweepTimer.Start();

      // Perform the local hypersweep
      hyperSweeper.LocalHyperSweep();

#ifdef DEBUG_PRINT_HYPER_SWEEPER
      {
        std::stringstream debugStream;
        debugStream << "Local Hypersweep Complete" << std::endl;
        vtkm::worklet::contourtree_augmented::PrintHeader(b->IntrinsicVolume.GetNumberOfValues(),
                                                          debugStream);
        vtkm::cont::ArrayHandle<vtkm::Id> whichTreeSupernodeRegularIDs;
        // we copy the HCT information to a temp array because b->HierarchicalContourTree is a const
        vtkm::cont::ArrayHandle<vtkm::Id> hctGRIds;
        vtkm::cont::ArrayHandle<vtkm::Id> hctSupernodes;
        vtkm::cont::Algorithm::Copy(b->HierarchicalContourTree.RegularNodeGlobalIds, hctGRIds);
        vtkm::cont::Algorithm::Copy(b->HierarchicalContourTree.Supernodes, hctSupernodes);
        vtkm::worklet::contourtree_augmented::PermuteArrayWithMaskedIndex<vtkm::Id>(
          hctGRIds, hctSupernodes, whichTreeSupernodeRegularIDs);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Regular ID", whichTreeSupernodeRegularIDs, debugStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Intrinsic", b->IntrinsicVolume, -1, debugStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Dependent", b->DependentVolume, -1, debugStream);
        VTKM_LOG_S(vtkm::cont::LogLevel::Info, debugStream.str());
      }
#endif

      // Log the local hypersweep time
      localHypersweepTimingsStream << "    Local Hypersweep (block=" << b->LocalBlockNo
                                   << ") : " << localHypersweepTimer.GetElapsedTime() << " seconds"
                                   << std::endl;
      localHypersweepTimer.Start();

      // Log the timing stats we collected
      VTKM_LOG_S(this->TimingsLogLevel,
                 std::endl
                   << "    ------------ Compute Local Hypersweep (block=" << b->LocalBlockNo
                   << ")  ------------" << std::endl
                   << localHypersweepTimingsStream.str());
    });

  // Log time for performing the local hypersweep
  timingsStream << "    " << std::setw(38) << std::left << "Compute Local Hypersweep"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // Reduce
  // partners for merge over regular block grid
  vtkmdiy::reduce(
    hierarchical_hyper_sweep_master,
    assigner,
    partners,
    vtkm::worklet::contourtree_distributed::CobmineHyperSweepBlockFunctor<FieldType>{});

  // Log time to merge hypersweep results
  timingsStream << "    " << std::setw(38) << std::left << "Merge Hypersweep Results"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // Add the intrinsic and dependent volumes to the output vectors
  intrinsicVolumes.resize(inputContourTreeMaster.size());
  dependentVolumes.resize(inputContourTreeMaster.size());
  hierarchical_hyper_sweep_master.foreach (
    [&](HyperSweepBlock* b, const vtkmdiy::Master::ProxyWithLink&) {
      intrinsicVolumes[b->LocalBlockNo] = b->IntrinsicVolume;
      dependentVolumes[b->LocalBlockNo] = b->DependentVolume;

#ifdef DEBUG_PRINT_HIERARCHICAL_AUGMENTER
      {
        VTKM_LOG_S(this->TreeLogLevel, "Block " << b->GlobalBlockId);
        /*VTKM_LOG_S(
          this->TreeLogLevel,
          b->HierarchicalContourTree.DebugPrint("Called from DumpVolumes", __FILE__, __LINE__));*/
        std::ostringstream volumeStream;
        vtkm::worklet::contourtree_augmented::PrintHeader(b->IntrinsicVolume.GetNumberOfValues(),
                                                          volumeStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Intrinsic Volume", b->IntrinsicVolume, -1, volumeStream);
        vtkm::worklet::contourtree_augmented::PrintIndices(
          "Dependent Volume", b->DependentVolume, -1, volumeStream);
        VTKM_LOG_S(this->TreeLogLevel, volumeStream.str());
      }
#endif
    });
}


//-----------------------------------------------------------------------------
template <typename FieldType>
VTKM_CONT void ContourTreeUniformDistributed::DoPostExecute(
  const vtkm::cont::PartitionedDataSet& input,
  vtkm::cont::PartitionedDataSet& result)
{
  vtkm::cont::Timer timer;
  timer.Start();
  std::stringstream timingsStream;

  auto comm = vtkm::cont::EnvironmentTracker::GetCommunicator();
  vtkm::Id size = comm.size();
  vtkm::Id rank = comm.rank();

  // ******** 1. Fan in to compute the hiearchical contour tree ********
  // 1.1 Setup DIY to do global binary reduction of neighbouring blocks.
  // See also RecuctionOperation struct for example

  // 1.1.1 Create the vtkmdiy master ...
  using DistributedContourTreeBlockData =
    vtkm::worklet::contourtree_distributed::DistributedContourTreeBlockData<FieldType>;
  vtkmdiy::Master master(comm,
                         1,  // Use 1 thread, VTK-M will do the treading
                         -1, // All blocks in memory
                         0,  // No create function (since all blocks in memory)
                         DistributedContourTreeBlockData::Destroy);

  // ... and record time for creating the DIY master
  timingsStream << "    " << std::setw(38) << std::left
                << "Create DIY Master (Distributed Contour Tree)"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // 1.1.2 Compute the gids for our local blocks
  using RegularDecomposer = vtkmdiy::RegularDecomposer<vtkmdiy::DiscreteBounds>;

  RegularDecomposer::DivisionsVector diyDivisions;
  std::vector<int> vtkmdiyLocalBlockGids;
  vtkmdiy::DiscreteBounds diyBounds(0);
  if (this->BlocksPerDimension[0] == -1)
  {
    VTKM_LOG_S(vtkm::cont::LogLevel::Info,
               "BlocksPerDimension not set. Computing block indices "
               "from information in CellSetStructured.");
    diyBounds = vtkm::filter::scalar_topology::internal::ComputeBlockIndices(
      input, diyDivisions, vtkmdiyLocalBlockGids);

    // Set BlocksPerDimension fromn diyDivisions result as add them
    // as information to the output data set for use in subsequent
    // filters
    this->BlocksPerDimension = vtkm::Id3{ 1, 1, 1 };
    for (unsigned int d = 0; d < diyDivisions.size(); ++d)
    {
      this->BlocksPerDimension[d] = diyDivisions[d];
    }
  }
  else
  {
    VTKM_LOG_S(vtkm::cont::LogLevel::Info,
               "BlocksPerDimension set. Using information provided by caller.");
    diyBounds =
      vtkm::filter::scalar_topology::internal::ComputeBlockIndices(input,
                                                                   this->BlocksPerDimension,
                                                                   this->LocalBlockIndices,
                                                                   diyDivisions,
                                                                   vtkmdiyLocalBlockGids);
  }
  int numDims = diyBounds.min.dimension();
  int globalNumberOfBlocks =
    std::accumulate(diyDivisions.cbegin(), diyDivisions.cend(), 1, std::multiplies<int>{});

  // Record time to compute the local block ids
  timingsStream << "    " << std::setw(38) << std::left << "Compute Block Ids and Local Links"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // 1.1.3 Setup the block data for DIY and add it to master
  // Note: globalPointDimensions is defined outside the loop since it is needed later.
  // It may be set multiple times in the loop, but always to the same value.
  vtkm::Id3 globalPointDimensions;
  for (vtkm::Id bi = 0; bi < input.GetNumberOfPartitions(); bi++)
  {
    // Get the input block and associated cell set information
    auto currBlock = input.GetPartition(static_cast<vtkm::Id>(bi));
    vtkm::Id3 pointDimensions, globalPointIndexStart;
    currBlock.GetCellSet().CastAndCallForTypes<VTKM_DEFAULT_CELL_SET_LIST_STRUCTURED>(
      vtkm::worklet::contourtree_augmented::GetLocalAndGlobalPointDimensions(),
      pointDimensions,
      globalPointDimensions,
      globalPointIndexStart);

    // Create the local data block structure and set extents
    auto newBlock = new DistributedContourTreeBlockData();

    // Copy global block id into the local data block for use in the hierarchical augmentation
    newBlock->GlobalBlockId = vtkmdiyLocalBlockGids[bi];
    newBlock->LocalBlockNo = bi;
    newBlock->BlockOrigin = globalPointIndexStart;
    newBlock->BlockSize = pointDimensions;
    newBlock->FixedBlockOrigin = globalPointIndexStart;
    newBlock->FixedBlockSize = pointDimensions;

    // Save local tree information for fan out; TODO/FIXME: Try to avoid copy
    newBlock->ContourTrees.push_back(this->LocalContourTrees[bi]);
    newBlock->InteriorForests.push_back(this->LocalInteriorForests[bi]);

    // ... Compute arrays needed for constructing contour tree mesh
    const auto sortOrder = this->LocalMeshes[bi].SortOrder;
    // ... Compute the global mesh index for the partially augmented contour tree. I.e., here we
    // don't need the global mesh index for all nodes, but only for the augmented nodes from the
    // tree. We, hence, permute the sortOrder by contourTree.augmentednodes and then compute the
    // GlobalMeshIndex by tranforming those indices with our IdRelabler
    vtkm::worklet::contourtree_augmented::IdArrayType localGlobalMeshIndex;
    vtkm::cont::ArrayHandlePermutation<vtkm::worklet::contourtree_augmented::IdArrayType,
                                       vtkm::worklet::contourtree_augmented::IdArrayType>
      permutedSortOrder(this->LocalBoundaryTrees[bi].VertexIndex, sortOrder);
    auto transformedIndex = vtkm::cont::make_ArrayHandleTransform(
      permutedSortOrder,
      vtkm::worklet::contourtree_augmented::mesh_dem::IdRelabeler(
        globalPointIndexStart, pointDimensions, globalPointDimensions));
    vtkm::cont::Algorithm::Copy(transformedIndex, localGlobalMeshIndex);

    // ... get data values
    auto currField =
      currBlock.GetField(this->GetActiveFieldName(), this->GetActiveFieldAssociation());
    vtkm::cont::ArrayHandle<FieldType> fieldData;
    vtkm::cont::ArrayCopy(currField.GetData(), fieldData);

    // ... compute and store the actual mesh
    newBlock->ContourTreeMeshes.emplace_back(this->LocalBoundaryTrees[bi].VertexIndex,
                                             this->LocalBoundaryTrees[bi].Superarcs,
                                             sortOrder,
                                             fieldData,
                                             localGlobalMeshIndex);

    // NOTE: Use dummy link to make DIY happy. The dummy link is never used, since all
    //       communication is via RegularDecomposer, which sets up its own links. No need
    //       to keep the pointer, as DIY will "own" it and delete it when no longer needed.
    // NOTE: Since we passed a "Destroy" function to DIY master, it will own the local data
    //       blocks and delete them when done.
    master.add(vtkmdiyLocalBlockGids[bi], newBlock, new vtkmdiy::Link);
  } // for

  // Record time for computing block data and adding it to master
  timingsStream << "    " << std::setw(38) << std::left
                << "Computing Block Data for Fan In and Adding Data Blocks to DIY"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // ... save for debugging in text and .gv/.dot format. We could do this in the loop above,
  //     but in order to separate timing we do this here and the extra loop over the partitions
  //     should not be significnatly more expensive then doing it all in one loop
  if (this->SaveDotFiles)
  {
    master.foreach ([&](DistributedContourTreeBlockData* b, const vtkmdiy::Master::ProxyWithLink&) {
      // save the contour tree mesh
      std::string contourTreeMeshFileName = std::string("Rank_") +
        std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
        std::to_string(static_cast<int>(b->LocalBlockNo)) +
        std::string("_Initial_Step_3_BRACT_Mesh.txt");
      b->ContourTreeMeshes.back().Save(contourTreeMeshFileName.c_str());

      // save the corresponding .gv file
      std::string boundaryTreeMeshFileName = std::string("Rank_") +
        std::to_string(static_cast<int>(rank)) + std::string("_Block_") +
        std::to_string(static_cast<int>(b->LocalBlockNo)) +
        std::string("_Initial_Step_5_BRACT_Mesh.gv");
      std::ofstream boundaryTreeMeshFile(boundaryTreeMeshFileName);
      boundaryTreeMeshFile
        << vtkm::worklet::contourtree_distributed::ContourTreeMeshDotGraphPrint<FieldType>(
             std::string("Block ") + std::to_string(static_cast<int>(rank)) +
               std::string(" Initial Step 5 BRACT Mesh"),
             b->ContourTreeMeshes.back(),
             worklet::contourtree_distributed::SHOW_CONTOUR_TREE_MESH_ALL);
    }); // master.for_each

    // Record time for saving debug data
    timingsStream << "    " << std::setw(38) << std::left << "Save block data for debug"
                  << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
    timer.Start();
  } // if(SaveDotFiles)

  // 1.2 Set up DIY for binary reduction
  // 1.2.1 Define the decomposition of the domain into regular blocks
  RegularDecomposer::BoolVector shareFace(3, true);
  RegularDecomposer::BoolVector wrap(3, false);
  RegularDecomposer::CoordinateVector ghosts(3, 1);
  RegularDecomposer decomposer(static_cast<int>(numDims),
                               diyBounds,
                               globalNumberOfBlocks,
                               shareFace,
                               wrap,
                               ghosts,
                               diyDivisions);

  // Define which blocks live on which rank so that vtkmdiy can manage them
  vtkmdiy::DynamicAssigner assigner(comm, static_cast<int>(size), globalNumberOfBlocks);
  for (vtkm::Id bi = 0; bi < input.GetNumberOfPartitions(); bi++)
  {
    assigner.set_rank(static_cast<int>(rank), vtkmdiyLocalBlockGids[static_cast<size_t>(bi)]);
  }

  // Record time for creating the decomposer and assigner
  timingsStream << "    " << std::setw(38) << std::left << "Create DIY Decomposer and Assigner"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // 1.2.2  Fix the vtkmdiy links.
  vtkmdiy::fix_links(master, assigner);

  // Record time to fix the links
  timingsStream << "    " << std::setw(38) << std::left
                << "Fix DIY Links (Distributed Contour Tree)"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // partners for merge over regular block grid
  vtkmdiy::RegularSwapPartners partners(
    decomposer, // domain decomposition
    2,          // radix of k-ary reduction.
    true        // contiguous: true=distance doubling, false=distance halving
  );

  // Record time to create the swap partners
  timingsStream << "    " << std::setw(38) << std::left << "Create DIY Swap Partners"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();
  // 1.3 Perform fan-in reduction
  const vtkm::worklet::contourtree_distributed::ComputeDistributedContourTreeFunctor<FieldType>
    computeDistributedContourTreeFunctor(globalPointDimensions,
                                         this->UseBoundaryExtremaOnly,
                                         this->TimingsLogLevel,
                                         this->TreeLogLevel);
  vtkmdiy::reduce(master, assigner, partners, computeDistributedContourTreeFunctor);
  // Record timing for the actual reduction
  timingsStream << "    " << std::setw(38) << std::left << "Fan In Reduction"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // Be safe! that the Fan In is completed on all blocks and ranks
  comm.barrier();

  timingsStream << "    " << std::setw(38) << std::left << "Post Fan In Barrier"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // ******** 2. Fan out to update all the tree ********
  master.foreach ([&](DistributedContourTreeBlockData* blockData,
                      const vtkmdiy::Master::ProxyWithLink&) {
#ifdef DEBUG_PRINT_CTUD
    // Save the contour tree, contour tree meshes, and interior forest data for debugging
    vtkm::filter::contourtree_distributed_detail::SaveAfterFanInResults(
      blockData, rank, this->TreeLogLevel);
#endif
    vtkm::cont::Timer iterationTimer;
    iterationTimer.Start();
    std::stringstream fanoutTimingsStream;

    // Fan out
    auto nRounds = blockData->ContourTrees.size() - 1;

    blockData->HierarchicalTree.Initialize(static_cast<vtkm::Id>(nRounds),
                                           blockData->ContourTrees[nRounds],
                                           blockData->ContourTreeMeshes[nRounds - 1]);

    // save the corresponding .gv file
    if (this->SaveDotFiles)
    {
      vtkm::filter::scalar_topology::contourtree_distributed_detail::SaveHierarchicalTreeDot(
        blockData, rank, nRounds);
    } // if(this->SaveDotFiles)

    fanoutTimingsStream << "    Fan Out Init Hierarchical Tree (block=" << blockData->LocalBlockNo
                        << ") : " << iterationTimer.GetElapsedTime() << " seconds" << std::endl;
    iterationTimer.Start();

    for (auto round = nRounds - 1; round > 0; round--)
    {
      iterationTimer.Start();
      vtkm::worklet::contourtree_distributed::
        TreeGrafter<vtkm::worklet::contourtree_augmented::ContourTreeMesh<FieldType>, FieldType>
          grafter(&(blockData->ContourTreeMeshes[round - 1]),
                  blockData->ContourTrees[round],
                  &(blockData->InteriorForests[round]));
      grafter.GraftInteriorForests(static_cast<vtkm::Id>(round),
                                   blockData->HierarchicalTree,
                                   blockData->ContourTreeMeshes[round - 1].SortedValues);
      // save the corresponding .gv file
      if (this->SaveDotFiles)
      {
        vtkm::filter::scalar_topology::contourtree_distributed_detail::SaveHierarchicalTreeDot(
          blockData, rank, nRounds);
      } // if(this->SaveDotFiles)
      // Log the time for each of the iterations of the fan out loop
      fanoutTimingsStream << "    Fan Out Time (block=" << blockData->LocalBlockNo
                          << " , round=" << round << ") : " << iterationTimer.GetElapsedTime()
                          << " seconds" << std::endl;
    } // for

    // bottom level
    iterationTimer.Start();
    vtkm::worklet::contourtree_distributed::
      TreeGrafter<vtkm::worklet::contourtree_augmented::DataSetMesh, FieldType>
        grafter(&(this->LocalMeshes[static_cast<std::size_t>(blockData->LocalBlockNo)]),
                blockData->ContourTrees[0],
                &(blockData->InteriorForests[0]));
    vtkm::cont::DataSet currBlock = input.GetPartition(blockData->LocalBlockNo);
    auto currField =
      currBlock.GetField(this->GetActiveFieldName(), this->GetActiveFieldAssociation());
    vtkm::cont::ArrayHandle<FieldType> fieldData;
    vtkm::cont::ArrayCopy(currField.GetData(), fieldData);

    vtkm::Id3 pointDimensions, globalPointIndexStart;
    // globalPointDimensions already defined in parent scope
    currBlock.GetCellSet().CastAndCallForTypes<VTKM_DEFAULT_CELL_SET_LIST_STRUCTURED>(
      vtkm::worklet::contourtree_augmented::GetLocalAndGlobalPointDimensions(),
      pointDimensions,
      globalPointDimensions,
      globalPointIndexStart);

    auto localToGlobalIdRelabeler = vtkm::worklet::contourtree_augmented::mesh_dem::IdRelabeler(
      globalPointIndexStart, pointDimensions, globalPointDimensions);
    grafter.GraftInteriorForests(
      0, blockData->HierarchicalTree, fieldData, &localToGlobalIdRelabeler);

    // Log the time for each of the iterations of the fan out loop
    fanoutTimingsStream << "    Fan Out Time (block=" << blockData->LocalBlockNo << " , round=" << 0
                        << ") : " << iterationTimer.GetElapsedTime() << " seconds" << std::endl;

    // Log the timing stats we collected
    VTKM_LOG_S(this->TimingsLogLevel,
               std::endl
                 << "    ------------ Fan Out (block=" << blockData->LocalBlockNo
                 << ")  ------------" << std::endl
                 << fanoutTimingsStream.str());
  });

  // 2.2 Log timings for fan out
  timingsStream << "    " << std::setw(38) << std::left << "Fan Out Foreach"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  // Add a barrier to make the interpretation of timings easier. In this way ranks that
  // finish early in the Fan Out wait here rather than waiting later some time during augmentation
  // or in post execute where we can't easily measure the impact of this wait. Adding the
  // barrier should not have a significant impact on performance as the wait would happen later on anyways.
  comm.barrier();
  timingsStream << "    " << std::setw(38) << std::left << "Post Fan Out Barrier"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();


  // Compute the volume for pre-simplification if we want to pre-simplify
  // The dependent volumes from the unaugemented hierarchical tree are used for the pre-simplification
  // as part of HierarchicalAugmenter.Initialize.
  std::vector<vtkm::cont::ArrayHandle<vtkm::Id>> unaugmentedDependentVolumes;
  if (this->PresimplifyThreshold > 0)
  {
    // we don't need the unaugemented intrinsic Volumes for the pre-simplification, so we
    // use a local variable that is being deleted automatically after the context
    std::vector<vtkm::cont::ArrayHandle<vtkm::Id>> unaugmentedIntrinsicVolumes;
    // Compute the volume for the base hierarchical tree before augmentation in order to allow for pre-simplification.
    this->ComputeVolumeMetric(
      master,
      assigner,
      partners,
      FieldType{},
      timingsStream,
      input,
      false, //   use the unaugmented hierarchical tree (i.e., the base tree) for the volume computation
      unaugmentedIntrinsicVolumes,
      unaugmentedDependentVolumes);
    timingsStream << "    " << std::setw(38) << std::left << "Compute Volume for Presimplication"
                  << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
    timer.Start();
  }

  // ******** 3. Augment the hierarchical tree if requested ********
  if (this->AugmentHierarchicalTree)
  {
    vtkm::Id localPresimplifyThreshold = this->PresimplifyThreshold;
    master.foreach ([globalPointDimensions, localPresimplifyThreshold, unaugmentedDependentVolumes](
                      DistributedContourTreeBlockData* blockData,
                      const vtkmdiy::Master::ProxyWithLink&) {
      // if we don't presimplify then use a NULL pointer for the dependent volume used for pre-simplification
      vtkm::worklet::contourtree_augmented::IdArrayType* volumeArrayForPresimplifiction = NULL;
      // if we presimplify then get a pointer for the dependent volume for the current block
      if (localPresimplifyThreshold > 0)
      {
        volumeArrayForPresimplifiction =
          const_cast<vtkm::worklet::contourtree_augmented::IdArrayType*>(
            &unaugmentedDependentVolumes[blockData->LocalBlockNo]);
      }
      // Initialize the hierarchical augmenter
      blockData->HierarchicalAugmenter.Initialize(
        blockData->GlobalBlockId,
        &blockData->HierarchicalTree,
        &blockData->AugmentedTree,
        blockData->FixedBlockOrigin, // Origin of the data block
        blockData->FixedBlockSize,   // Extends of the data block
        globalPointDimensions,       // global point dimensions
        volumeArrayForPresimplifiction, // DependentVolume if we computed it or NULL if no presimplification is used
        localPresimplifyThreshold // presimplify if threshold is > 0
      );
    });

    timingsStream << "    " << std::setw(38) << std::left << "Initalize Hierarchical Trees"
                  << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
    timer.Start();

    vtkmdiy::reduce(master,
                    assigner,
                    partners,
                    vtkm::worklet::contourtree_distributed::HierarchicalAugmenterFunctor<FieldType>{
                      this->TimingsLogLevel });

    // Clear all swap data as it is no longer needed
    master.foreach (
      [](DistributedContourTreeBlockData* blockData, const vtkmdiy::Master::ProxyWithLink&) {
        blockData->HierarchicalAugmenter.ReleaseSwapArrays();
      });

    timingsStream << "    " << std::setw(38) << std::left << "Compute/Exchange Attachment Points"
                  << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
    timer.Start();

    master.foreach (
      [](DistributedContourTreeBlockData* blockData, const vtkmdiy::Master::ProxyWithLink&) {
        blockData->HierarchicalAugmenter.BuildAugmentedTree();
      });

    timingsStream << "    " << std::setw(38) << std::left << "Build Augmented Tree"
                  << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
    timer.Start();
  }

  // ******** 4. Create output data set ********
  std::vector<vtkm::cont::DataSet> hierarchicalTreeOutputDataSet(master.size());
  master.foreach ([&](DistributedContourTreeBlockData* blockData,
                      const vtkmdiy::Master::ProxyWithLink&) {
    std::stringstream createOutdataTimingsStream;
    vtkm::cont::Timer iterationTimer;
    iterationTimer.Start();

    // Use the augmented tree if available or otherwise use the unaugmented hierarchical tree from the current block
    const auto& blockHierarchcialTree = this->AugmentHierarchicalTree
      ? (*blockData->HierarchicalAugmenter.AugmentedTree)
      : blockData->HierarchicalTree;

    // Add the information to the output data set
    blockHierarchcialTree.AddToVTKMDataSet(hierarchicalTreeOutputDataSet[blockData->LocalBlockNo]);

    // Save information required to set up DIY
    vtkm::cont::ArrayHandle<vtkm::Id> vtkmGlobalBlockIdAH;
    vtkmGlobalBlockIdAH.Allocate(1);
    auto vtkmGlobalBlockIdWP = vtkmGlobalBlockIdAH.WritePortal();
    vtkmGlobalBlockIdWP.Set(0, blockData->GlobalBlockId);
    vtkm::cont::Field vtkmGlobalBlockIdField(
      "vtkmGlobalBlockId", vtkm::cont::Field::Association::WholeDataSet, vtkmGlobalBlockIdAH);
    hierarchicalTreeOutputDataSet[blockData->LocalBlockNo].AddField(vtkmGlobalBlockIdField);
    vtkm::cont::ArrayHandle<vtkm::Id> vtkmBlocksPerDimensionAH;
    vtkmBlocksPerDimensionAH.Allocate(3);
    auto vtkmBlocksPerDimensionWP = vtkmBlocksPerDimensionAH.WritePortal();
    vtkmBlocksPerDimensionWP.Set(0, this->BlocksPerDimension[0]);
    vtkmBlocksPerDimensionWP.Set(1, this->BlocksPerDimension[1]);
    vtkmBlocksPerDimensionWP.Set(2, this->BlocksPerDimension[2]);
    vtkm::cont::Field vtkmBlocksPerDimensionField("vtkmBlocksPerDimension",
                                                  vtkm::cont::Field::Association::WholeDataSet,
                                                  vtkmBlocksPerDimensionAH);
    hierarchicalTreeOutputDataSet[blockData->LocalBlockNo].AddField(vtkmBlocksPerDimensionField);

    // Copy cell set from input data set. This is mainly to ensure that the output data set
    // has a defined cell set. Without one, serialization for DIY does not work properly.
    // Having the extents of the input data set may also help in other use cases.
    // For example, the ComputeVolume method gets information from this cell set as will
    // the branch decomposition filter.
    hierarchicalTreeOutputDataSet[blockData->LocalBlockNo].SetCellSet(
      input.GetPartition(blockData->LocalBlockNo).GetCellSet());

    // Log the time for each of the iterations of the fan out loop
    createOutdataTimingsStream << "    Create Output Dataset (block=" << blockData->LocalBlockNo
                               << ") : " << iterationTimer.GetElapsedTime() << " seconds"
                               << std::endl;
    iterationTimer.Start();

    // save the corresponding .gv file
    if (this->SaveDotFiles)
    {
      auto nRounds = blockData->ContourTrees.size() - 1;
      vtkm::filter::scalar_topology::contourtree_distributed_detail::SaveHierarchicalTreeDot(
        blockData, rank, nRounds);

      createOutdataTimingsStream << "    Save Dot (block=" << blockData->LocalBlockNo
                                 << ") : " << iterationTimer.GetElapsedTime() << " seconds"
                                 << std::endl;
      iterationTimer.Start();
    } // if(this->SaveDotFiles)

    // Log the timing stats we collected
    VTKM_LOG_S(this->TimingsLogLevel,
               std::endl
                 << "    ------------ Create Output Data (block=" << blockData->LocalBlockNo
                 << ")  ------------" << std::endl
                 << createOutdataTimingsStream.str());

    // Log the stats from the hierarchical contour tree
    VTKM_LOG_S(this->TreeLogLevel,
               std::endl
                 << "    ------------ Hierarchical Tree Construction Stats ------------"
                 << std::endl
                 << std::setw(42) << std::left << "    LocalBlockNo"
                 << ": " << blockData->LocalBlockNo << std::endl
                 << blockData->HierarchicalTree.PrintTreeStats() << std::endl);
  }); // master.foreach

  // Log total tree computation and augmentation time
  timingsStream << "    " << std::setw(38) << std::left << "Create Output Data"
                << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
  timer.Start();

  if (this->AugmentHierarchicalTree)
  {
    std::vector<vtkm::cont::ArrayHandle<vtkm::Id>> augmentedIntrinsicVolumes;
    std::vector<vtkm::cont::ArrayHandle<vtkm::Id>> augmentedDependentVolumes;
    this->ComputeVolumeMetric(master,
                              assigner,
                              partners,
                              FieldType{},
                              timingsStream,
                              input,
                              true, //   use the augmented tree
                              augmentedIntrinsicVolumes,
                              augmentedDependentVolumes);
    timer.Start();

    master.foreach (
      [&](DistributedContourTreeBlockData* blockData, const vtkmdiy::Master::ProxyWithLink&) {
        // Add the intrinsic and dependent volumes to the output data set
        vtkm::cont::Field intrinsicVolumeField("IntrinsicVolume",
                                               vtkm::cont::Field::Association::WholeDataSet,
                                               augmentedIntrinsicVolumes[blockData->LocalBlockNo]);
        hierarchicalTreeOutputDataSet[blockData->LocalBlockNo].AddField(intrinsicVolumeField);
        vtkm::cont::Field dependentVolumeField("DependentVolume",
                                               vtkm::cont::Field::Association::WholeDataSet,
                                               augmentedDependentVolumes[blockData->LocalBlockNo]);
        hierarchicalTreeOutputDataSet[blockData->LocalBlockNo].AddField(dependentVolumeField);
        // Log the time for adding hypersweep data to the output dataset
        timingsStream << "    " << std::setw(38) << std::left << "Add Volume Output Data"
                      << ": " << timer.GetElapsedTime() << " seconds" << std::endl;
      });
  }

  VTKM_LOG_S(this->TimingsLogLevel,
             std::endl
               << "    ------------ DoPostExecute Timings ------------" << std::endl
               << timingsStream.str());

  result = vtkm::cont::PartitionedDataSet(hierarchicalTreeOutputDataSet);
} // DoPostExecute

} // namespace scalar_topology
} // namespace filter
} // namespace vtkm::filter
