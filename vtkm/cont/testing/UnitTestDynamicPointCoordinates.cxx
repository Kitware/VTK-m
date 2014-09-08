//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014. Los Alamos National Security
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#include <vtkm/cont/DynamicPointCoordinates.h>

#include <vtkm/cont/StorageImplicit.h>
#include <vtkm/cont/StorageListTag.h>

#include <vtkm/cont/internal/IteratorFromArrayPortal.h>

#include <vtkm/cont/testing/Testing.h>

#include <vector>

namespace {

const vtkm::Extent3 EXTENT = vtkm::Extent3(vtkm::Id3(0,0,0), vtkm::Id3(9,9,9));
const vtkm::Vector3 ORIGIN = vtkm::Vector3(0, 0, 0);
const vtkm::Vector3 SPACING = vtkm::Vector3(1, 1, 1);

const vtkm::Id3 DIMENSION = vtkm::ExtentPointDimensions(EXTENT);
const vtkm::Id ARRAY_SIZE = DIMENSION[0]*DIMENSION[1]*DIMENSION[2];

vtkm::Vector3 TestValue(vtkm::Id index)
{
  vtkm::Id3 index3d = vtkm::ExtentPointFlatIndexToTopologyIndex(index, EXTENT);
  return vtkm::Vector3(vtkm::Scalar(index3d[0]),
                       vtkm::Scalar(index3d[1]),
                       vtkm::Scalar(index3d[2]));
}

int g_CheckArrayInvocations;

struct CheckArray
{
  CheckArray() {
    g_CheckArrayInvocations = 0;
  }

  template<typename Storage>
  void operator()(
      const vtkm::cont::ArrayHandle<vtkm::Vector3,Storage> &array) const
  {
    std::cout << "    In CastAndCall functor" << std::endl;
    g_CheckArrayInvocations++;
    typename vtkm::cont::ArrayHandle<vtkm::Vector3,Storage>::PortalConstControl portal =
        array.GetPortalConstControl();

    VTKM_TEST_ASSERT(portal.GetNumberOfValues() == ARRAY_SIZE,
                     "Array has wrong number of values.");

    for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
    {
      const vtkm::Vector3 receivedValue = portal.Get(index);
      const vtkm::Vector3 expectedValue = TestValue(index);
      VTKM_TEST_ASSERT(receivedValue == expectedValue,
                       "Got bad value in array.");
    }
  }
};

struct UnusualPortal
{
  typedef vtkm::Vector3 ValueType;

  VTKM_EXEC_CONT_EXPORT
  vtkm::Id GetNumberOfValues() const { return ARRAY_SIZE; }

  VTKM_EXEC_CONT_EXPORT
  ValueType Get(vtkm::Id index) const { return TestValue(index); }
};

class ArrayHandleWithUnusualStorage
    : public vtkm::cont::ArrayHandle<vtkm::Vector3, vtkm::cont::StorageTagImplicit<UnusualPortal> >
{
  typedef vtkm::cont::ArrayHandle<vtkm::Vector3, vtkm::cont::StorageTagImplicit<UnusualPortal> >
      Superclass;
public:
  VTKM_CONT_EXPORT
  ArrayHandleWithUnusualStorage()
    : Superclass(Superclass::PortalConstControl()) {  }
};

struct StorageListTagUnusual :
    vtkm::ListTagBase<ArrayHandleWithUnusualStorage::StorageTag>
{  };

struct PointCoordinatesUnusual : vtkm::cont::internal::PointCoordinatesBase
{
  template<typename Functor, typename TypeList, typename StorageList>
  void CastAndCall(const Functor &f, TypeList, StorageList) const
  {
    f(ArrayHandleWithUnusualStorage());
  }
};

struct PointCoordinatesListUnusual
    : vtkm::ListTagBase<PointCoordinatesUnusual> {  };

void TryDefaultArray()
{
  std::cout << "Trying a basic point coordinates array with a default storage."
            << std::endl;
  std::vector<vtkm::Vector3> buffer(ARRAY_SIZE);
  for (vtkm::Id index = 0; index < ARRAY_SIZE; index++)
  {
    buffer[index] = TestValue(index);
  }

  vtkm::cont::DynamicPointCoordinates pointCoordinates =
      vtkm::cont::DynamicPointCoordinates(
        vtkm::cont::make_ArrayHandle(buffer));

  pointCoordinates.CastAndCall(CheckArray());

  VTKM_TEST_ASSERT(g_CheckArrayInvocations == 1,
                   "CastAndCall functor not called expected number of times.");
}

void TryUnusualStorage()
{
  std::cout << "Trying a basic point coordinates array with an unusual storage."
               << std::endl;

  vtkm::cont::DynamicPointCoordinates pointCoordinates =
      vtkm::cont::DynamicPointCoordinates(
        vtkm::cont::PointCoordinatesArray(ArrayHandleWithUnusualStorage()));

  std::cout << "  Make sure we get an exception when we can't find the type."
            << std::endl;
  try
  {
    pointCoordinates.CastAndCall(CheckArray());
    VTKM_TEST_FAIL("CastAndCall failed to error for unrecognized storage.");
  }
  catch (vtkm::cont::ErrorControlBadValue error)
  {
    std::cout << "  Caught expected exception for unrecognized storage: "
              << std::endl << "    " << error.GetMessage() << std::endl;
  }

  std::cout << "  Recast storage and try again." << std::endl;
  pointCoordinates.ResetStorageList(StorageListTagUnusual())
      .CastAndCall(CheckArray());
  VTKM_TEST_ASSERT(g_CheckArrayInvocations == 1,
                   "CastAndCall functor not called expected number of times.");
}

void TryUniformPointCoordinates()
{
  std::cout << "Trying uniform point coordinates." << std::endl;

  vtkm::cont::DynamicPointCoordinates pointCoordinates =
      vtkm::cont::DynamicPointCoordinates(
        vtkm::cont::PointCoordinatesUniform(EXTENT, ORIGIN, SPACING));

  pointCoordinates.CastAndCall(CheckArray());

  VTKM_TEST_ASSERT(g_CheckArrayInvocations == 1,
                   "CastAndCall functor not called expected number of times.");
}

void TryUnusualPointCoordinates()
{
  std::cout << "Trying an unusual point coordinates object." << std::endl;

  vtkm::cont::DynamicPointCoordinates pointCoordinates =
      vtkm::cont::DynamicPointCoordinates(PointCoordinatesUnusual());

  std::cout << "  Make sure we get an exception when we can't find the type."
            << std::endl;
  try
  {
    pointCoordinates.CastAndCall(CheckArray());
    VTKM_TEST_FAIL("CastAndCall failed to error for unrecognized storage.");
  }
  catch (vtkm::cont::ErrorControlBadValue error)
  {
    std::cout << "  Caught expected exception for unrecognized point coordinates: "
              << std::endl << "    " << error.GetMessage() << std::endl;
  }

  std::cout << "  Recast storage and try again." << std::endl;
  pointCoordinates.ResetPointCoordinatesList(PointCoordinatesListUnusual())
      .CastAndCall(CheckArray());
  VTKM_TEST_ASSERT(g_CheckArrayInvocations == 1,
                   "CastAndCall functor not called expected number of times.");
}

void DynamicPointCoordiantesTest()
{
  TryDefaultArray();
  TryUnusualStorage();
  TryUniformPointCoordinates();
  TryUnusualPointCoordinates();
}

} // anonymous namespace

int UnitTestDynamicPointCoordinates(int, char *[])
{
  return vtkm::cont::testing::Testing::Run(DynamicPointCoordiantesTest);
}
