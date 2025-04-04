//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/testing/MakeTestDataSet.h>
#include <vtkm/cont/testing/Testing.h>

#include <vtkm/filter/clean_grid/CleanGrid.h>
#include <vtkm/filter/entity_extraction/ExternalFaces.h>

using vtkm::cont::testing::MakeTestDataSet;

namespace
{

// convert a 5x5x5 uniform grid to unstructured grid
vtkm::cont::DataSet MakeDataTestSet1()
{
  vtkm::cont::DataSet ds = MakeTestDataSet().Make3DUniformDataSet1();

  vtkm::filter::clean_grid::CleanGrid clean;
  clean.SetCompactPointFields(false);
  clean.SetMergePoints(false);
  return clean.Execute(ds);
}

vtkm::cont::DataSet MakeDataTestSet2()
{
  return MakeTestDataSet().Make3DExplicitDataSet5();
}

vtkm::cont::DataSet MakeDataTestSet3()
{
  return MakeTestDataSet().Make3DUniformDataSet1();
}

vtkm::cont::DataSet MakeDataTestSet4()
{
  return MakeTestDataSet().Make3DRectilinearDataSet0();
}

vtkm::cont::DataSet MakeDataTestSet5()
{
  return MakeTestDataSet().Make3DExplicitDataSet6();
}

vtkm::cont::DataSet MakeUniformDataTestSet()
{
  return MakeTestDataSet().Make3DUniformDataSet1();
}

vtkm::cont::DataSet MakeCurvilinearDataTestSet()
{
  vtkm::cont::DataSet data = MakeUniformDataTestSet();
  vtkm::cont::ArrayHandle<vtkm::Vec3f> coords;
  vtkm::cont::CoordinateSystem oldCoords = data.GetCoordinateSystem();
  vtkm::cont::ArrayCopy(oldCoords.GetData(), coords);
  data.AddCoordinateSystem(oldCoords.GetName(), coords);
  return data;
}

void TestExternalFacesExplicitGrid(const vtkm::cont::DataSet& ds,
                                   bool compactPoints,
                                   vtkm::Id numExpectedExtFaces,
                                   vtkm::Id numExpectedPoints = 0,
                                   bool passPolyData = true)
{
  //Run the External Faces filter
  vtkm::filter::entity_extraction::ExternalFaces externalFaces;
  externalFaces.SetCompactPoints(compactPoints);
  externalFaces.SetPassPolyData(passPolyData);
  vtkm::cont::DataSet resultds = externalFaces.Execute(ds);

  // verify cellset
  const vtkm::Id numOutputExtFaces = resultds.GetNumberOfCells();
  VTKM_TEST_ASSERT(numOutputExtFaces == numExpectedExtFaces, "Number of External Faces mismatch");

  // verify fields
  VTKM_TEST_ASSERT(resultds.HasField("pointvar"), "Point field not mapped successfully");
  VTKM_TEST_ASSERT(resultds.HasField("cellvar"), "Cell field not mapped successfully");

  // verify CompactPoints
  if (compactPoints)
  {
    vtkm::Id numOutputPoints = resultds.GetCoordinateSystem(0).GetNumberOfPoints();
    VTKM_TEST_ASSERT(numOutputPoints == numExpectedPoints,
                     "Incorrect number of points after compacting");
  }
}

void TestWithHexahedraMesh()
{
  std::cout << "Testing with Hexahedra mesh\n";
  vtkm::cont::DataSet ds = MakeDataTestSet1();
  std::cout << "Compact Points Off\n";
  TestExternalFacesExplicitGrid(ds, false, 96); // 4x4 * 6 = 96
  std::cout << "Compact Points On\n";
  TestExternalFacesExplicitGrid(ds, true, 96, 98); // 5x5x5 - 3x3x3 = 98
}

void TestWithHeterogeneousMesh()
{
  std::cout << "Testing with Heterogeneous mesh\n";
  vtkm::cont::DataSet ds = MakeDataTestSet2();
  std::cout << "Compact Points Off\n";
  TestExternalFacesExplicitGrid(ds, false, 12);
  std::cout << "Compact Points On\n";
  TestExternalFacesExplicitGrid(ds, true, 12, 11);
}

void TestWithUniformMesh()
{
  std::cout << "Testing with Uniform mesh\n";
  vtkm::cont::DataSet ds = MakeDataTestSet3();
  std::cout << "Compact Points Off\n";
  TestExternalFacesExplicitGrid(ds, false, 16 * 6);
  std::cout << "Compact Points On\n";
  TestExternalFacesExplicitGrid(ds, true, 16 * 6, 98);
}

void TestWithRectilinearMesh()
{
  std::cout << "Testing with Rectilinear mesh\n";
  vtkm::cont::DataSet ds = MakeDataTestSet4();
  std::cout << "Compact Points Off\n";
  TestExternalFacesExplicitGrid(ds, false, 16);
  std::cout << "Compact Points On\n";
  TestExternalFacesExplicitGrid(ds, true, 16, 18);
}

void TestWithMixed2Dand3DMesh()
{
  std::cout << "Testing with mixed poly data and 3D mesh\n";
  vtkm::cont::DataSet ds = MakeDataTestSet5();
  std::cout << "Compact Points Off, Pass Poly Data On\n";
  TestExternalFacesExplicitGrid(ds, false, 12);
  std::cout << "Compact Points On, Pass Poly Data On\n";
  TestExternalFacesExplicitGrid(ds, true, 12, 8);
  std::cout << "Compact Points Off, Pass Poly Data Off\n";
  TestExternalFacesExplicitGrid(ds, false, 6, 8, false);
  std::cout << "Compact Points On, Pass Poly Data Off\n";
  TestExternalFacesExplicitGrid(ds, true, 6, 5, false);
}

void TestExternalFacesStructuredGrid(const vtkm::cont::DataSet& ds, bool compactPoints)
{
  // Get the dimensions of the grid.
  vtkm::cont::CellSetStructured<3> cellSet;
  ds.GetCellSet().AsCellSet(cellSet);
  vtkm::Id3 pointDims = cellSet.GetPointDimensions();
  vtkm::Id3 cellDims = cellSet.GetCellDimensions();

  //Run the External Faces filter
  vtkm::filter::entity_extraction::ExternalFaces externalFaces;
  externalFaces.SetCompactPoints(compactPoints);
  vtkm::cont::DataSet resultds = externalFaces.Execute(ds);

  // verify cellset
  vtkm::Id numExpectedExtFaces = ((2 * cellDims[0] * cellDims[1]) + // x-y faces
                                  (2 * cellDims[0] * cellDims[2]) + // x-z faces
                                  (2 * cellDims[1] * cellDims[2])); // y-z faces
  const vtkm::Id numOutputExtFaces = resultds.GetNumberOfCells();
  VTKM_TEST_ASSERT(numOutputExtFaces == numExpectedExtFaces, "Number of External Faces mismatch");

  // verify fields
  VTKM_TEST_ASSERT(resultds.HasField("pointvar"), "Point field not mapped successfully");
  VTKM_TEST_ASSERT(resultds.HasField("cellvar"), "Cell field not mapped successfully");

  // verify CompactPoints
  if (compactPoints)
  {
    vtkm::Id numExpectedPoints = ((2 * pointDims[0] * pointDims[1])   // x-y faces
                                  + (2 * pointDims[0] * pointDims[2]) // x-z faces
                                  + (2 * pointDims[1] * pointDims[2]) // y-z faces
                                  - (4 * pointDims[0])                // overcounted x edges
                                  - (4 * pointDims[1])                // overcounted y edges
                                  - (4 * pointDims[2])                // overcounted z edges
                                  + 8);                               // undercounted corners
    vtkm::Id numOutputPoints = resultds.GetNumberOfPoints();
    VTKM_TEST_ASSERT(numOutputPoints == numExpectedPoints);
  }
  else
  {
    VTKM_TEST_ASSERT(resultds.GetNumberOfPoints() == ds.GetNumberOfPoints());
  }
}

void TestWithUniformGrid()
{
  std::cout << "Testing with uniform grid\n";
  vtkm::cont::DataSet ds = MakeUniformDataTestSet();
  std::cout << "Compact Points Off\n";
  TestExternalFacesStructuredGrid(ds, false);
  std::cout << "Compact Points On\n";
  TestExternalFacesStructuredGrid(ds, true);
}

void TestWithCurvilinearGrid()
{
  std::cout << "Testing with curvilinear grid\n";
  vtkm::cont::DataSet ds = MakeCurvilinearDataTestSet();
  std::cout << "Compact Points Off\n";
  TestExternalFacesStructuredGrid(ds, false);
  std::cout << "Compact Points On\n";
  TestExternalFacesStructuredGrid(ds, true);
}

void TestExternalFacesFilter()
{
  TestWithHeterogeneousMesh();
  TestWithHexahedraMesh();
  TestWithUniformMesh();
  TestWithRectilinearMesh();
  TestWithMixed2Dand3DMesh();
  TestWithUniformGrid();
  TestWithCurvilinearGrid();
}

} // anonymous namespace

int UnitTestExternalFacesFilter(int argc, char* argv[])
{
  return vtkm::cont::testing::Testing::Run(TestExternalFacesFilter, argc, argv);
}
