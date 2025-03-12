//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include "TestingFlow.h"

#include <vtkm/cont/EnvironmentTracker.h>

#include <vtkm/cont/testing/Testing.h>

namespace
{

void DoTest()
{
  auto comm = vtkm::cont::EnvironmentTracker::GetCommunicator();

  FilterType filterType = PATHLINE;

  for (vtkm::Id nPerRank = 1; nPerRank < 3; ++nPerRank)
  {
    for (bool useGhost : { true, false })
    {
      for (bool useThreaded : { true, false })
      {
        for (bool useBlockIds : { true, false })
        {
          //Run blockIds with and without block duplication.
          if (useBlockIds && comm.size() > 1)
          {
            TestPartitionedDataSet(nPerRank, useGhost, filterType, useThreaded, useBlockIds, false);
            TestPartitionedDataSet(nPerRank, useGhost, filterType, useThreaded, useBlockIds, true);
          }
          else
          {
            TestPartitionedDataSet(nPerRank, useGhost, filterType, useThreaded, useBlockIds, false);
          }
        }
      }
    }
  }
}

} // anonymous namespace

int UnitTestPathlineMPI(int argc, char* argv[])
{
  return vtkm::cont::testing::Testing::Run(DoTest, argc, argv);
}
