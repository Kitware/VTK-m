//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include <vtkm/cont/testing/MakeTestDataSet.h>
#include <vtkm/cont/testing/Testing.h>

#include <vtkm/filter/entity_extraction/ExtractStructured.h>

using vtkm::cont::testing::MakeTestDataSet;

namespace
{

class TestingExtractStructured
{
public:
  static void TestUniform2D()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make2DUniformDataSet1();

    // Change point index start from 0, 0 to 10, 14
    vtkm::cont::CellSetStructured<2> cellSet;
    dataset.GetCellSet().AsCellSet(cellSet);
    cellSet.SetGlobalPointDimensions(vtkm::Id2{ 15, 19 });
    cellSet.SetGlobalPointIndexStart(vtkm::Id2{ 10, 14 });
    dataset.SetCellSet(cellSet);
    dataset.PrintSummary(std::cout);

    vtkm::RangeId3 range(1, 4, 1, 4, 0, 1);
    vtkm::Id3 sample(1, 1, 1);

    vtkm::filter::entity_extraction::ExtractStructured extract;
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 9),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 4),
                     "Wrong result for ExtractStructured worklet");
    vtkm::cont::CellSetStructured<2> outputCellSet;
    output.GetCellSet().AsCellSet(outputCellSet);
    VTKM_TEST_ASSERT(test_equal(outputCellSet.GetGlobalPointIndexStart()[0], 11),
                     "Wrong result for ExtractStructured PointIndexStart");
    VTKM_TEST_ASSERT(test_equal(outputCellSet.GetGlobalPointIndexStart()[1], 15),
                     "Wrong result for ExtractStructured PointIndexStart");
    VTKM_TEST_ASSERT(test_equal(outputCellSet.GetGlobalPointDimensions()[0], 15),
                     "Wrong result for ExtractStructured GlobalPointDimensions");
    VTKM_TEST_ASSERT(test_equal(outputCellSet.GetGlobalPointDimensions()[1], 19),
                     "Wrong result for ExtractStructured GlobalPointDimensions");


    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 71.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(8) == 91.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 5.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(3) == 10.0f, "Wrong cell field data");
  }

  static void TestUniform3D0()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // VOI within dataset
    extract.SetVOI(1, 4, 1, 4, 1, 4);
    extract.SetSampleRate(1, 1, 1);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 27),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 8),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 99.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(26) == 97.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 21.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(7) == 42.0f, "Wrong cell field data");
  }

  static void TestUniform3D1()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // VOI surrounds dataset
    vtkm::Id3 minPoint(-1, -1, -1);
    vtkm::Id3 maxPoint(8, 8, 8);
    extract.SetVOI(minPoint, maxPoint);
    extract.SetSampleRate(1, 1, 1);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 125),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 64),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(31) == 99.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(93) == 97.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 0.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(63) == 63.0f, "Wrong cell field data");
  }

  static void TestUniform3D2()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();
    vtkm::filter::entity_extraction::ExtractStructured extract;

    // VOI surrounds dataset
    vtkm::RangeId3 range(-1, 3, -1, 3, -1, 3);
    vtkm::Id3 sample(1, 1, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 27),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 8),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(26) == 15.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 0.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(7) == 21.0f, "Wrong cell field data");
  }

  static void TestUniform3D3()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();
    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 intersects dataset on far boundary
    vtkm::RangeId3 range(1, 8, 1, 8, 1, 8);
    vtkm::Id3 sample(1, 1, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 64),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 27),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 99.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(63) == 0.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 21.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(26) == 63.0f, "Wrong cell field data");
  }

  static void TestUniform3D4()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 intersects dataset without corner
    vtkm::RangeId3 range(2, 8, 1, 4, 1, 4);
    vtkm::Id3 sample(1, 1, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 27),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 8),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 90.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(26) == 0.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 22.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(7) == 43.0f, "Wrong cell field data");
  }

  static void TestUniform3D5()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 intersects dataset with plane
    vtkm::RangeId3 range(2, 8, 1, 2, 1, 4);
    vtkm::Id3 sample(1, 1, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 9),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 4),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 90.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(8) == 0.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 22.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(3) == 39.0f, "Wrong cell field data");
  }

  static void TestUniform3D6()
  {
    std::cout << "Testing extract structured uniform" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 within data set with sampling
    vtkm::RangeId3 range(0, 5, 0, 5, 1, 4);
    vtkm::Id3 sample(2, 2, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 27),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 8),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(26) == 0.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 16.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(3) == 26.0f, "Wrong cell field data");
  }

  static void TestUniform3D7()
  {
    std::cout << "Testing extract structured uniform, exclude boundary" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 within data set with sampling
    vtkm::RangeId3 range(0, 5, 0, 5, 1, 4);
    vtkm::Id3 sample(3, 3, 2);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);
    extract.SetIncludeBoundary(false); // default

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 8),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 1),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(3) == 99.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(4) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(7) == 97.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 16.0f, "Wrong cell field data");
  }

  static void TestUniform3D8()
  {
    std::cout << "Testing extract structured uniform, include boundary" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DUniformDataSet1();
    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 within data set with sampling
    vtkm::RangeId3 range(0, 5, 0, 5, 1, 4);
    vtkm::Id3 sample(3, 3, 2);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);
    extract.SetIncludeBoundary(true);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 18),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 4),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(4) == 99.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(5) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(7) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(13) == 97.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 16.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(1) == 19.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(2) == 28.0f, "Wrong cell field data");
    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(3) == 31.0f, "Wrong cell field data");
  }

  static void TestRectilinear2D()
  {
    std::cout << "Testing extract structured rectilinear" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make2DRectilinearDataSet0();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3
    vtkm::RangeId3 range(0, 2, 0, 2, 0, 1);
    vtkm::Id3 sample(1, 1, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 4),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 1),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(3) == 4.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 0.0f, "Wrong cell field data");
  }

  static void TestRectilinear3D()
  {
    std::cout << "Testing extract structured rectilinear" << std::endl;
    vtkm::cont::DataSet dataset = MakeTestDataSet().Make3DRectilinearDataSet0();

    vtkm::filter::entity_extraction::ExtractStructured extract;

    // RangeId3 and subsample
    vtkm::RangeId3 range(0, 2, 0, 2, 0, 2);
    vtkm::Id3 sample(1, 1, 1);
    extract.SetVOI(range);
    extract.SetSampleRate(sample);

    extract.SetFieldsToPass({ "pointvar", "cellvar" });
    vtkm::cont::DataSet output = extract.Execute(dataset);
    VTKM_TEST_ASSERT(test_equal(output.GetCellSet().GetNumberOfPoints(), 8),
                     "Wrong result for ExtractStructured worklet");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), 1),
                     "Wrong result for ExtractStructured worklet");

    vtkm::cont::ArrayHandle<vtkm::Float32> outPointData;
    vtkm::cont::ArrayHandle<vtkm::Float32> outCellData;
    output.GetField("pointvar").GetData().AsArrayHandle(outPointData);
    output.GetField("cellvar").GetData().AsArrayHandle(outCellData);

    VTKM_TEST_ASSERT(
      test_equal(output.GetCellSet().GetNumberOfPoints(), outPointData.GetNumberOfValues()),
      "Data/Geometry mismatch for ExtractStructured filter");
    VTKM_TEST_ASSERT(test_equal(output.GetNumberOfCells(), outCellData.GetNumberOfValues()),
                     "Data/Geometry mismatch for ExtractStructured filter");

    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(0) == 0.0f, "Wrong point field data");
    VTKM_TEST_ASSERT(outPointData.ReadPortal().Get(7) == 10.0f, "Wrong point field data");

    VTKM_TEST_ASSERT(outCellData.ReadPortal().Get(0) == 0.0f, "Wrong cell field data");
  }

  void operator()() const
  {
    TestingExtractStructured::TestUniform2D();
    TestingExtractStructured::TestUniform3D0();
    TestingExtractStructured::TestUniform3D1();
    TestingExtractStructured::TestUniform3D2();
    TestingExtractStructured::TestUniform3D3();
    TestingExtractStructured::TestUniform3D4();
    TestingExtractStructured::TestUniform3D5();
    TestingExtractStructured::TestUniform3D6();
    TestingExtractStructured::TestUniform3D7();
    TestingExtractStructured::TestUniform3D8();
    TestingExtractStructured::TestRectilinear2D();
    TestingExtractStructured::TestRectilinear3D();
  }
};
}

int UnitTestExtractStructuredFilter(int argc, char* argv[])
{
  return vtkm::cont::testing::Testing::Run(TestingExtractStructured(), argc, argv);
}
