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

#include <mpi.h>
static void InitDebug()
{
  int nRanks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << rank << " InitDebug()" << std::endl;

  for (int r = 0; r < nRanks; r++)
  {
    if (r == rank)
      std::cout << "Rank: " << r << " pid= " << getpid() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int doLoop = (rank == 0);
  while (doLoop)
  {
    sleep(1);
    //std::cout<<"Sleeping..."<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
    std::cout << "Ready to go!" << std::endl;
}


namespace
{

void DoTest()
{
  auto comm = vtkm::cont::EnvironmentTracker::GetCommunicator();
  //InitDebug();


  FilterType filterType = PARTICLE_ADVECTION;

  for (vtkm::Id nPerRank = 1; nPerRank < 3; ++nPerRank)
  {
    for (bool useGhost : { true, false })
    {
      if (useGhost)
        continue;
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

int UnitTestAdvectionMPI(int argc, char* argv[])
{
  return vtkm::cont::testing::Testing::Run(DoTest, argc, argv);
}
