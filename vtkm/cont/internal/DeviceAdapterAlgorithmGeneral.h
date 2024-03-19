//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#ifndef vtk_m_cont_internal_DeviceAdapterAlgorithmGeneral_h
#define vtk_m_cont_internal_DeviceAdapterAlgorithmGeneral_h

#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleDecorator.h>
#include <vtkm/cont/ArrayHandleDiscard.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/ArrayHandleView.h>
#include <vtkm/cont/ArrayHandleZip.h>
#include <vtkm/cont/BitField.h>
#include <vtkm/cont/Logging.h>
#include <vtkm/cont/internal/FunctorsGeneral.h>
#include <vtkm/cont/internal/Hints.h>

#include <vtkm/exec/internal/ErrorMessageBuffer.h>
#include <vtkm/exec/internal/TaskSingular.h>

#include <vtkm/BinaryPredicates.h>
#include <vtkm/TypeTraits.h>

#include <vtkm/internal/Windows.h>

#include <type_traits>

namespace vtkm
{
namespace cont
{
namespace internal
{

/// \brief General implementations of device adapter algorithms.
///
/// This struct provides algorithms that implement "general" device adapter
/// algorithms. If a device adapter provides implementations for Schedule,
/// and Synchronize, the rest of the algorithms can be implemented by calling
/// these functions.
///
/// It should be noted that we recommend that you also implement Sort,
/// ScanInclusive, and ScanExclusive for improved performance.
///
/// An easy way to implement the DeviceAdapterAlgorithm specialization is to
/// subclass this and override the implementation of methods as necessary.
/// As an example, the code would look something like this.
///
/// \code{.cpp}
/// template<>
/// struct DeviceAdapterAlgorithm<DeviceAdapterTagFoo>
///    : DeviceAdapterAlgorithmGeneral<DeviceAdapterAlgorithm<DeviceAdapterTagFoo>,
///                                    DeviceAdapterTagFoo>
/// {
///   template<typename Hints, typename Functor>
///   VTKM_CONT static void Schedule(Hints, Functor functor, vtkm::Id numInstances)
///   {
///     ...
///   }
///
///   template<typename Functor>
///   VTKM_CONT static void Schedule(Functor&& functor, vtkm::Id numInstances)
///   {
///     Schedule(vtkm::cont::internal::HintList<>{}, functor, numInstances);
///   }
///
///   template<typename Hints, typename Functor>
///   VTKM_CONT static void Schedule(Hints, Functor functor, vtkm::Id3 maxRange)
///   {
///     ...
///   }
///
///   template<typename Functor>
///   VTKM_CONT static void Schedule(Functor&& functor, vtkm::Id3 maxRange)
///   {
///     Schedule(vtkm::cont::internal::HintList<>{}, functor, numInstances);
///   }
///
///   VTKM_CONT static void Synchronize()
///   {
///     ...
///   }
/// };
/// \endcode
///
/// You might note that DeviceAdapterAlgorithmGeneral has two template
/// parameters that are redundant. Although the first parameter, the class for
/// the actual DeviceAdapterAlgorithm class containing Schedule, and
/// Synchronize is the same as DeviceAdapterAlgorithm<DeviceAdapterTag>, it is
/// made a separate template parameter to avoid a recursive dependence between
/// DeviceAdapterAlgorithmGeneral.h and DeviceAdapterAlgorithm.h
///
template <class DerivedAlgorithm, class DeviceAdapterTag>
struct DeviceAdapterAlgorithmGeneral
{
  //--------------------------------------------------------------------------
  // Get Execution Value
  // This method is used internally to get a single element from the execution
  // array. Normally you would just use ArrayGetValue, but that functionality
  // relies on the device adapter algorithm and would create a circular
  // dependency.
private:
  template <typename T, class CIn>
  VTKM_CONT static T GetExecutionValue(const vtkm::cont::ArrayHandle<T, CIn>& input, vtkm::Id index)
  {
    vtkm::cont::ArrayHandle<T, vtkm::cont::StorageTagBasic> output;

    {
      vtkm::cont::Token token;

      auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
      auto outputPortal = output.PrepareForOutput(1, DeviceAdapterTag(), token);

      CopyKernel<decltype(inputPortal), decltype(outputPortal)> kernel(
        inputPortal, outputPortal, index);

      DerivedAlgorithm::Schedule(kernel, 1);
    }

    return output.ReadPortal().Get(0);
  }

public:
  //--------------------------------------------------------------------------
  // BitFieldToUnorderedSet
  template <typename IndicesStorage>
  VTKM_CONT static vtkm::Id BitFieldToUnorderedSet(
    const vtkm::cont::BitField& bits,
    vtkm::cont::ArrayHandle<Id, IndicesStorage>& indices)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id numBits = bits.GetNumberOfBits();

    vtkm::cont::Token token;

    auto bitsPortal = bits.PrepareForInput(DeviceAdapterTag{}, token);
    auto indicesPortal = indices.PrepareForOutput(numBits, DeviceAdapterTag{}, token);

    std::atomic<vtkm::UInt64> popCount;
    popCount.store(0, std::memory_order_seq_cst);

    using Functor = BitFieldToUnorderedSetFunctor<decltype(bitsPortal), decltype(indicesPortal)>;
    Functor functor{ bitsPortal, indicesPortal, popCount };

    DerivedAlgorithm::Schedule(functor, functor.GetNumberOfInstances());
    DerivedAlgorithm::Synchronize();

    token.DetachFromAll();

    numBits = static_cast<vtkm::Id>(popCount.load(std::memory_order_seq_cst));

    indices.Allocate(numBits, vtkm::CopyFlag::On);
    return numBits;
  }

  //--------------------------------------------------------------------------
  // Copy
  template <typename T, typename U, class CIn, class COut>
  VTKM_CONT static void Copy(const vtkm::cont::ArrayHandle<T, CIn>& input,
                             vtkm::cont::ArrayHandle<U, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::cont::Token token;

    const vtkm::Id inSize = input.GetNumberOfValues();
    auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(inSize, DeviceAdapterTag(), token);

    CopyKernel<decltype(inputPortal), decltype(outputPortal)> kernel(inputPortal, outputPortal);
    DerivedAlgorithm::Schedule(kernel, inSize);
  }

  //--------------------------------------------------------------------------
  // CopyIf
  template <typename T, typename U, class CIn, class CStencil, class COut, class UnaryPredicate>
  VTKM_CONT static void CopyIf(const vtkm::cont::ArrayHandle<T, CIn>& input,
                               const vtkm::cont::ArrayHandle<U, CStencil>& stencil,
                               vtkm::cont::ArrayHandle<T, COut>& output,
                               UnaryPredicate unary_predicate)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    VTKM_ASSERT(input.GetNumberOfValues() == stencil.GetNumberOfValues());
    vtkm::Id arrayLength = stencil.GetNumberOfValues();

    using IndexArrayType = vtkm::cont::ArrayHandle<vtkm::Id, vtkm::cont::StorageTagBasic>;
    IndexArrayType indices;

    {
      vtkm::cont::Token token;

      auto stencilPortal = stencil.PrepareForInput(DeviceAdapterTag(), token);
      auto indexPortal = indices.PrepareForOutput(arrayLength, DeviceAdapterTag(), token);

      StencilToIndexFlagKernel<decltype(stencilPortal), decltype(indexPortal), UnaryPredicate>
        indexKernel(stencilPortal, indexPortal, unary_predicate);

      DerivedAlgorithm::Schedule(indexKernel, arrayLength);
    }

    vtkm::Id outArrayLength = DerivedAlgorithm::ScanExclusive(indices, indices);

    {
      vtkm::cont::Token token;

      auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
      auto stencilPortal = stencil.PrepareForInput(DeviceAdapterTag(), token);
      auto indexPortal = indices.PrepareForOutput(arrayLength, DeviceAdapterTag(), token);
      auto outputPortal = output.PrepareForOutput(outArrayLength, DeviceAdapterTag(), token);

      CopyIfKernel<decltype(inputPortal),
                   decltype(stencilPortal),
                   decltype(indexPortal),
                   decltype(outputPortal),
                   UnaryPredicate>
        copyKernel(inputPortal, stencilPortal, indexPortal, outputPortal, unary_predicate);
      DerivedAlgorithm::Schedule(copyKernel, arrayLength);
    }
  }

  template <typename T, typename U, class CIn, class CStencil, class COut>
  VTKM_CONT static void CopyIf(const vtkm::cont::ArrayHandle<T, CIn>& input,
                               const vtkm::cont::ArrayHandle<U, CStencil>& stencil,
                               vtkm::cont::ArrayHandle<T, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    ::vtkm::NotZeroInitialized unary_predicate;
    DerivedAlgorithm::CopyIf(input, stencil, output, unary_predicate);
  }

  //--------------------------------------------------------------------------
  // CopySubRange
  template <typename T, typename U, class CIn, class COut>
  VTKM_CONT static bool CopySubRange(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                     vtkm::Id inputStartIndex,
                                     vtkm::Id numberOfElementsToCopy,
                                     vtkm::cont::ArrayHandle<U, COut>& output,
                                     vtkm::Id outputIndex = 0)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    const vtkm::Id inSize = input.GetNumberOfValues();

    // Check if the ranges overlap and fail if they do.
    if (input == output &&
        ((outputIndex >= inputStartIndex &&
          outputIndex < inputStartIndex + numberOfElementsToCopy) ||
         (inputStartIndex >= outputIndex &&
          inputStartIndex < outputIndex + numberOfElementsToCopy)))
    {
      return false;
    }

    if (inputStartIndex < 0 || numberOfElementsToCopy < 0 || outputIndex < 0 ||
        inputStartIndex >= inSize)
    { //invalid parameters
      return false;
    }

    //determine if the numberOfElementsToCopy needs to be reduced
    if (inSize < (inputStartIndex + numberOfElementsToCopy))
    { //adjust the size
      numberOfElementsToCopy = (inSize - inputStartIndex);
    }

    const vtkm::Id outSize = output.GetNumberOfValues();
    const vtkm::Id copyOutEnd = outputIndex + numberOfElementsToCopy;
    if (outSize < copyOutEnd)
    { //output is not large enough
      if (outSize == 0)
      { //since output has nothing, just need to allocate to correct length
        output.Allocate(copyOutEnd);
      }
      else
      { //we currently have data in this array, so preserve it in the new
        //resized array
        vtkm::cont::ArrayHandle<U, COut> temp;
        temp.Allocate(copyOutEnd);
        DerivedAlgorithm::CopySubRange(output, 0, outSize, temp);
        output = temp;
      }
    }

    vtkm::cont::Token token;

    auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForInPlace(DeviceAdapterTag(), token);

    CopyKernel<decltype(inputPortal), decltype(outputPortal)> kernel(
      inputPortal, outputPortal, inputStartIndex, outputIndex);
    DerivedAlgorithm::Schedule(kernel, numberOfElementsToCopy);
    return true;
  }

  //--------------------------------------------------------------------------
  // Count Set Bits
  VTKM_CONT static vtkm::Id CountSetBits(const vtkm::cont::BitField& bits)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::cont::Token token;

    auto bitsPortal = bits.PrepareForInput(DeviceAdapterTag{}, token);

    std::atomic<vtkm::UInt64> popCount;
    popCount.store(0, std::memory_order_relaxed);

    using Functor = CountSetBitsFunctor<decltype(bitsPortal)>;
    Functor functor{ bitsPortal, popCount };

    DerivedAlgorithm::Schedule(functor, functor.GetNumberOfInstances());
    DerivedAlgorithm::Synchronize();

    return static_cast<vtkm::Id>(popCount.load(std::memory_order_seq_cst));
  }

  //--------------------------------------------------------------------------
  // Fill Bit Field (bool, resize)
  VTKM_CONT static void Fill(vtkm::cont::BitField& bits, bool value, vtkm::Id numBits)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    if (numBits == 0)
    {
      bits.Allocate(0);
      return;
    }

    vtkm::cont::Token token;

    auto portal = bits.PrepareForOutput(numBits, DeviceAdapterTag{}, token);

    using WordType =
      typename vtkm::cont::BitField::template ExecutionTypes<DeviceAdapterTag>::WordTypePreferred;

    using Functor = FillBitFieldFunctor<decltype(portal), WordType>;
    Functor functor{ portal, value ? ~WordType{ 0 } : WordType{ 0 } };

    const vtkm::Id numWords = portal.template GetNumberOfWords<WordType>();
    DerivedAlgorithm::Schedule(functor, numWords);
  }

  //--------------------------------------------------------------------------
  // Fill Bit Field (bool)
  VTKM_CONT static void Fill(vtkm::cont::BitField& bits, bool value)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    const vtkm::Id numBits = bits.GetNumberOfBits();
    if (numBits == 0)
    {
      return;
    }

    vtkm::cont::Token token;

    auto portal = bits.PrepareForOutput(numBits, DeviceAdapterTag{}, token);

    using WordType =
      typename vtkm::cont::BitField::template ExecutionTypes<DeviceAdapterTag>::WordTypePreferred;

    using Functor = FillBitFieldFunctor<decltype(portal), WordType>;
    Functor functor{ portal, value ? ~WordType{ 0 } : WordType{ 0 } };

    const vtkm::Id numWords = portal.template GetNumberOfWords<WordType>();
    DerivedAlgorithm::Schedule(functor, numWords);
  }

  //--------------------------------------------------------------------------
  // Fill Bit Field (mask, resize)
  template <typename WordType>
  VTKM_CONT static void Fill(vtkm::cont::BitField& bits, WordType word, vtkm::Id numBits)
  {
    VTKM_STATIC_ASSERT_MSG(vtkm::cont::BitField::IsValidWordType<WordType>{}, "Invalid word type.");

    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    if (numBits == 0)
    {
      bits.Allocate(0);
      return;
    }

    vtkm::cont::Token token;

    auto portal = bits.PrepareForOutput(numBits, DeviceAdapterTag{}, token);

    // If less than 32 bits, repeat the word until we get a 32 bit pattern.
    // Using this for the pattern prevents races while writing small numbers
    // to adjacent memory locations.
    auto repWord = RepeatTo32BitsIfNeeded(word);
    using RepWordType = decltype(repWord);

    using Functor = FillBitFieldFunctor<decltype(portal), RepWordType>;
    Functor functor{ portal, repWord };

    const vtkm::Id numWords = portal.template GetNumberOfWords<RepWordType>();
    DerivedAlgorithm::Schedule(functor, numWords);
  }

  //--------------------------------------------------------------------------
  // Fill Bit Field (mask)
  template <typename WordType>
  VTKM_CONT static void Fill(vtkm::cont::BitField& bits, WordType word)
  {
    VTKM_STATIC_ASSERT_MSG(vtkm::cont::BitField::IsValidWordType<WordType>{}, "Invalid word type.");
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    const vtkm::Id numBits = bits.GetNumberOfBits();
    if (numBits == 0)
    {
      return;
    }

    vtkm::cont::Token token;

    auto portal = bits.PrepareForOutput(numBits, DeviceAdapterTag{}, token);

    // If less than 32 bits, repeat the word until we get a 32 bit pattern.
    // Using this for the pattern prevents races while writing small numbers
    // to adjacent memory locations.
    auto repWord = RepeatTo32BitsIfNeeded(word);
    using RepWordType = decltype(repWord);

    using Functor = FillBitFieldFunctor<decltype(portal), RepWordType>;
    Functor functor{ portal, repWord };

    const vtkm::Id numWords = portal.template GetNumberOfWords<RepWordType>();
    DerivedAlgorithm::Schedule(functor, numWords);
  }

  //--------------------------------------------------------------------------
  // Fill ArrayHandle
  template <typename T, typename S>
  VTKM_CONT static void Fill(vtkm::cont::ArrayHandle<T, S>& handle, const T& value)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    const vtkm::Id numValues = handle.GetNumberOfValues();
    if (numValues == 0)
    {
      return;
    }

    vtkm::cont::Token token;

    auto portal = handle.PrepareForOutput(numValues, DeviceAdapterTag{}, token);
    FillArrayHandleFunctor<decltype(portal)> functor{ portal, value };
    DerivedAlgorithm::Schedule(functor, numValues);
  }

  //--------------------------------------------------------------------------
  // Fill ArrayHandle (resize)
  template <typename T, typename S>
  VTKM_CONT static void Fill(vtkm::cont::ArrayHandle<T, S>& handle,
                             const T& value,
                             const vtkm::Id numValues)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);
    if (numValues == 0)
    {
      handle.ReleaseResources();
      return;
    }

    vtkm::cont::Token token;

    auto portal = handle.PrepareForOutput(numValues, DeviceAdapterTag{}, token);
    FillArrayHandleFunctor<decltype(portal)> functor{ portal, value };
    DerivedAlgorithm::Schedule(functor, numValues);
  }

  //--------------------------------------------------------------------------
  // Lower Bounds
  template <typename T, class CIn, class CVal, class COut>
  VTKM_CONT static void LowerBounds(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                    const vtkm::cont::ArrayHandle<T, CVal>& values,
                                    vtkm::cont::ArrayHandle<vtkm::Id, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id arraySize = values.GetNumberOfValues();

    vtkm::cont::Token token;

    auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
    auto valuesPortal = values.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(arraySize, DeviceAdapterTag(), token);

    LowerBoundsKernel<decltype(inputPortal), decltype(valuesPortal), decltype(outputPortal)> kernel(
      inputPortal, valuesPortal, outputPortal);

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template <typename T, class CIn, class CVal, class COut, class BinaryCompare>
  VTKM_CONT static void LowerBounds(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                    const vtkm::cont::ArrayHandle<T, CVal>& values,
                                    vtkm::cont::ArrayHandle<vtkm::Id, COut>& output,
                                    BinaryCompare binary_compare)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id arraySize = values.GetNumberOfValues();

    vtkm::cont::Token token;

    auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
    auto valuesPortal = values.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(arraySize, DeviceAdapterTag(), token);

    LowerBoundsComparisonKernel<decltype(inputPortal),
                                decltype(valuesPortal),
                                decltype(outputPortal),
                                BinaryCompare>
      kernel(inputPortal, valuesPortal, outputPortal, binary_compare);

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template <class CIn, class COut>
  VTKM_CONT static void LowerBounds(const vtkm::cont::ArrayHandle<vtkm::Id, CIn>& input,
                                    vtkm::cont::ArrayHandle<vtkm::Id, COut>& values_output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    DeviceAdapterAlgorithmGeneral<DerivedAlgorithm, DeviceAdapterTag>::LowerBounds(
      input, values_output, values_output);
  }

  //--------------------------------------------------------------------------
  // Reduce
#ifndef VTKM_CUDA
  // nvcc doesn't like the private class declaration so disable under CUDA
private:
#endif
  template <typename T, typename BinaryFunctor>
  class ReduceDecoratorImpl
  {
  public:
    VTKM_CONT ReduceDecoratorImpl() = default;

    VTKM_CONT
    ReduceDecoratorImpl(const T& initialValue, const BinaryFunctor& binaryFunctor)
      : InitialValue(initialValue)
      , ReduceOperator(binaryFunctor)
    {
    }

    template <typename Portal>
    VTKM_CONT ReduceKernel<Portal, T, BinaryFunctor> CreateFunctor(const Portal& portal) const
    {
      return ReduceKernel<Portal, T, BinaryFunctor>(
        portal, this->InitialValue, this->ReduceOperator);
    }

  private:
    T InitialValue;
    BinaryFunctor ReduceOperator;
  };

public:
  template <typename T, typename U, class CIn>
  VTKM_CONT static U Reduce(const vtkm::cont::ArrayHandle<T, CIn>& input, U initialValue)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    return DerivedAlgorithm::Reduce(input, initialValue, vtkm::Add());
  }

  template <typename T, typename U, class CIn, class BinaryFunctor>
  VTKM_CONT static U Reduce(const vtkm::cont::ArrayHandle<T, CIn>& input,
                            U initialValue,
                            BinaryFunctor binary_functor)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    //Crazy Idea:
    //We perform the reduction in two levels. The first level is performed by
    //an `ArrayHandleDecorator` which reduces 16 input values and maps them to
    //one value. The decorator array is then 1/16 the length of the input array,
    //and we can use inclusive scan as the second level to compute the final
    //result.
    vtkm::Id length = (input.GetNumberOfValues() / 16);
    length += (input.GetNumberOfValues() % 16 == 0) ? 0 : 1;
    auto reduced = vtkm::cont::make_ArrayHandleDecorator(
      length, ReduceDecoratorImpl<U, BinaryFunctor>(initialValue, binary_functor), input);

    vtkm::cont::ArrayHandle<U, vtkm::cont::StorageTagBasic> inclusiveScanStorage;
    const U scanResult =
      DerivedAlgorithm::ScanInclusive(reduced, inclusiveScanStorage, binary_functor);
    return scanResult;
  }

  //--------------------------------------------------------------------------
  // Reduce By Key
  template <typename T,
            typename U,
            class KIn,
            class VIn,
            class KOut,
            class VOut,
            class BinaryFunctor>
  VTKM_CONT static void ReduceByKey(const vtkm::cont::ArrayHandle<T, KIn>& keys,
                                    const vtkm::cont::ArrayHandle<U, VIn>& values,
                                    vtkm::cont::ArrayHandle<T, KOut>& keys_output,
                                    vtkm::cont::ArrayHandle<U, VOut>& values_output,
                                    BinaryFunctor binary_functor)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    using KeysOutputType = vtkm::cont::ArrayHandle<U, KOut>;

    VTKM_ASSERT(keys.GetNumberOfValues() == values.GetNumberOfValues());
    const vtkm::Id numberOfKeys = keys.GetNumberOfValues();

    if (numberOfKeys <= 1)
    { //we only have a single key/value so that is our output
      DerivedAlgorithm::Copy(keys, keys_output);
      DerivedAlgorithm::Copy(values, values_output);
      return;
    }

    //we need to determine based on the keys what is the keystate for
    //each key. The states are start, middle, end of a series and the special
    //state start and end of a series
    vtkm::cont::ArrayHandle<ReduceKeySeriesStates> keystate;

    {
      vtkm::cont::Token token;
      auto inputPortal = keys.PrepareForInput(DeviceAdapterTag(), token);
      auto keyStatePortal = keystate.PrepareForOutput(numberOfKeys, DeviceAdapterTag(), token);
      ReduceStencilGeneration<decltype(inputPortal), decltype(keyStatePortal)> kernel(
        inputPortal, keyStatePortal);
      DerivedAlgorithm::Schedule(kernel, numberOfKeys);
    }

    //next step is we need to reduce the values for each key. This is done
    //by running an inclusive scan over the values array using the stencil.
    //
    // this inclusive scan will write out two values, the first being
    // the value summed currently, the second being 0 or 1, with 1 being used
    // when this is a value of a key we need to write ( END or START_AND_END)
    {
      vtkm::cont::ArrayHandle<ReduceKeySeriesStates> stencil;
      vtkm::cont::ArrayHandle<U> reducedValues;

      auto scanInput = vtkm::cont::make_ArrayHandleZip(values, keystate);
      auto scanOutput = vtkm::cont::make_ArrayHandleZip(reducedValues, stencil);

      DerivedAlgorithm::ScanInclusive(
        scanInput, scanOutput, ReduceByKeyAdd<BinaryFunctor>(binary_functor));

      //at this point we are done with keystate, so free the memory
      keystate.ReleaseResources();

      // all we need know is an efficient way of doing the write back to the
      // reduced global memory. this is done by using CopyIf with the
      // stencil and values we just created with the inclusive scan
      DerivedAlgorithm::CopyIf(reducedValues, stencil, values_output, ReduceByKeyUnaryStencilOp());

    } //release all temporary memory

    // Don't bother with the keys_output if it's an ArrayHandleDiscard -- there
    // will be a runtime exception in Unique() otherwise:
    if (!vtkm::cont::IsArrayHandleDiscard<KeysOutputType>::value)
    {
      //find all the unique keys
      DerivedAlgorithm::Copy(keys, keys_output);
      DerivedAlgorithm::Unique(keys_output);
    }
  }

  //--------------------------------------------------------------------------
  // Scan Exclusive
  template <typename T, class CIn, class COut, class BinaryFunctor>
  VTKM_CONT static T ScanExclusive(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                   vtkm::cont::ArrayHandle<T, COut>& output,
                                   BinaryFunctor binaryFunctor,
                                   const T& initialValue)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id numValues = input.GetNumberOfValues();
    if (numValues <= 0)
    {
      output.ReleaseResources();
      return initialValue;
    }

    vtkm::cont::ArrayHandle<T, vtkm::cont::StorageTagBasic> inclusiveScan;
    T result = DerivedAlgorithm::ScanInclusive(input, inclusiveScan, binaryFunctor);

    vtkm::cont::Token token;

    auto inputPortal = inclusiveScan.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(numValues, DeviceAdapterTag(), token);

    InclusiveToExclusiveKernel<decltype(inputPortal), decltype(outputPortal), BinaryFunctor>
      inclusiveToExclusive(inputPortal, outputPortal, binaryFunctor, initialValue);

    DerivedAlgorithm::Schedule(inclusiveToExclusive, numValues);

    return binaryFunctor(initialValue, result);
  }

  template <typename T, class CIn, class COut>
  VTKM_CONT static T ScanExclusive(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                   vtkm::cont::ArrayHandle<T, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    return DerivedAlgorithm::ScanExclusive(
      input, output, vtkm::Sum(), vtkm::TypeTraits<T>::ZeroInitialization());
  }

  //--------------------------------------------------------------------------
  // Scan Exclusive Extend
  template <typename T, class CIn, class COut, class BinaryFunctor>
  VTKM_CONT static void ScanExtended(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                     vtkm::cont::ArrayHandle<T, COut>& output,
                                     BinaryFunctor binaryFunctor,
                                     const T& initialValue)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id numValues = input.GetNumberOfValues();
    if (numValues <= 0)
    {
      output.Allocate(1);
      output.WritePortal().Set(0, initialValue);
      return;
    }

    vtkm::cont::ArrayHandle<T, vtkm::cont::StorageTagBasic> inclusiveScan;
    T result = DerivedAlgorithm::ScanInclusive(input, inclusiveScan, binaryFunctor);

    vtkm::cont::Token token;

    auto inputPortal = inclusiveScan.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(numValues + 1, DeviceAdapterTag(), token);

    InclusiveToExtendedKernel<decltype(inputPortal), decltype(outputPortal), BinaryFunctor>
      inclusiveToExtended(inputPortal,
                          outputPortal,
                          binaryFunctor,
                          initialValue,
                          binaryFunctor(initialValue, result));

    DerivedAlgorithm::Schedule(inclusiveToExtended, numValues + 1);
  }

  template <typename T, class CIn, class COut>
  VTKM_CONT static void ScanExtended(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                     vtkm::cont::ArrayHandle<T, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    DerivedAlgorithm::ScanExtended(
      input, output, vtkm::Sum(), vtkm::TypeTraits<T>::ZeroInitialization());
  }

  //--------------------------------------------------------------------------
  // Scan Exclusive By Key
  template <typename KeyT,
            typename ValueT,
            typename KIn,
            typename VIn,
            typename VOut,
            class BinaryFunctor>
  VTKM_CONT static void ScanExclusiveByKey(const vtkm::cont::ArrayHandle<KeyT, KIn>& keys,
                                           const vtkm::cont::ArrayHandle<ValueT, VIn>& values,
                                           vtkm::cont::ArrayHandle<ValueT, VOut>& output,
                                           const ValueT& initialValue,
                                           BinaryFunctor binaryFunctor)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    VTKM_ASSERT(keys.GetNumberOfValues() == values.GetNumberOfValues());

    // 0. Special case for 0 and 1 element input
    vtkm::Id numberOfKeys = keys.GetNumberOfValues();

    if (numberOfKeys == 0)
    {
      return;
    }
    else if (numberOfKeys == 1)
    {
      output.Allocate(1);
      output.WritePortal().Set(0, initialValue);
      return;
    }

    // 1. Create head flags
    //we need to determine based on the keys what is the keystate for
    //each key. The states are start, middle, end of a series and the special
    //state start and end of a series
    vtkm::cont::ArrayHandle<ReduceKeySeriesStates> keystate;

    {
      vtkm::cont::Token token;
      auto inputPortal = keys.PrepareForInput(DeviceAdapterTag(), token);
      auto keyStatePortal = keystate.PrepareForOutput(numberOfKeys, DeviceAdapterTag(), token);
      ReduceStencilGeneration<decltype(inputPortal), decltype(keyStatePortal)> kernel(
        inputPortal, keyStatePortal);
      DerivedAlgorithm::Schedule(kernel, numberOfKeys);
    }

    // 2. Shift input and initialize elements at head flags position to initValue
    vtkm::cont::ArrayHandle<ValueT, vtkm::cont::StorageTagBasic> temp;
    {
      vtkm::cont::Token token;
      auto inputPortal = values.PrepareForInput(DeviceAdapterTag(), token);
      auto keyStatePortal = keystate.PrepareForInput(DeviceAdapterTag(), token);
      auto tempPortal = temp.PrepareForOutput(numberOfKeys, DeviceAdapterTag(), token);

      ShiftCopyAndInit<ValueT,
                       decltype(inputPortal),
                       decltype(keyStatePortal),
                       decltype(tempPortal)>
        kernel(inputPortal, keyStatePortal, tempPortal, initialValue);
      DerivedAlgorithm::Schedule(kernel, numberOfKeys);
    }
    // 3. Perform a ScanInclusiveByKey
    DerivedAlgorithm::ScanInclusiveByKey(keys, temp, output, binaryFunctor);
  }

  template <typename KeyT, typename ValueT, class KIn, typename VIn, typename VOut>
  VTKM_CONT static void ScanExclusiveByKey(const vtkm::cont::ArrayHandle<KeyT, KIn>& keys,
                                           const vtkm::cont::ArrayHandle<ValueT, VIn>& values,
                                           vtkm::cont::ArrayHandle<ValueT, VOut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    DerivedAlgorithm::ScanExclusiveByKey(
      keys, values, output, vtkm::TypeTraits<ValueT>::ZeroInitialization(), vtkm::Sum());
  }

  //--------------------------------------------------------------------------
  // Scan Inclusive
  template <typename T, class CIn, class COut>
  VTKM_CONT static T ScanInclusive(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                   vtkm::cont::ArrayHandle<T, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    return DerivedAlgorithm::ScanInclusive(input, output, vtkm::Add());
  }

private:
  template <typename T1, typename S1, typename T2, typename S2>
  VTKM_CONT static bool ArrayHandlesAreSame(const vtkm::cont::ArrayHandle<T1, S1>&,
                                            const vtkm::cont::ArrayHandle<T2, S2>&)
  {
    return false;
  }

  template <typename T, typename S>
  VTKM_CONT static bool ArrayHandlesAreSame(const vtkm::cont::ArrayHandle<T, S>& a1,
                                            const vtkm::cont::ArrayHandle<T, S>& a2)
  {
    return a1 == a2;
  }

public:
  template <typename T, class CIn, class COut, class BinaryFunctor>
  VTKM_CONT static T ScanInclusive(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                   vtkm::cont::ArrayHandle<T, COut>& output,
                                   BinaryFunctor binary_functor)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    if (!ArrayHandlesAreSame(input, output))
    {
      DerivedAlgorithm::Copy(input, output);
    }

    vtkm::Id numValues = output.GetNumberOfValues();
    if (numValues < 1)
    {
      return vtkm::TypeTraits<T>::ZeroInitialization();
    }

    {
      vtkm::cont::Token token;

      auto portal = output.PrepareForInPlace(DeviceAdapterTag(), token);
      using ScanKernelType = ScanKernel<decltype(portal), BinaryFunctor>;


      vtkm::Id stride;
      for (stride = 2; stride - 1 < numValues; stride *= 2)
      {
        ScanKernelType kernel(portal, binary_functor, stride, stride / 2 - 1);
        DerivedAlgorithm::Schedule(kernel, numValues / stride);
      }

      // Do reverse operation on odd indices. Start at stride we were just at.
      for (stride /= 2; stride > 1; stride /= 2)
      {
        ScanKernelType kernel(portal, binary_functor, stride, stride - 1);
        DerivedAlgorithm::Schedule(kernel, numValues / stride);
      }
    }

    return GetExecutionValue(output, numValues - 1);
  }

  template <typename KeyT, typename ValueT, class KIn, class VIn, class VOut>
  VTKM_CONT static void ScanInclusiveByKey(const vtkm::cont::ArrayHandle<KeyT, KIn>& keys,
                                           const vtkm::cont::ArrayHandle<ValueT, VIn>& values,
                                           vtkm::cont::ArrayHandle<ValueT, VOut>& values_output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    return DerivedAlgorithm::ScanInclusiveByKey(keys, values, values_output, vtkm::Add());
  }

  template <typename KeyT, typename ValueT, class KIn, class VIn, class VOut, class BinaryFunctor>
  VTKM_CONT static void ScanInclusiveByKey(const vtkm::cont::ArrayHandle<KeyT, KIn>& keys,
                                           const vtkm::cont::ArrayHandle<ValueT, VIn>& values,
                                           vtkm::cont::ArrayHandle<ValueT, VOut>& values_output,
                                           BinaryFunctor binary_functor)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    VTKM_ASSERT(keys.GetNumberOfValues() == values.GetNumberOfValues());
    const vtkm::Id numberOfKeys = keys.GetNumberOfValues();

    if (numberOfKeys <= 1)
    { //we only have a single key/value so that is our output
      DerivedAlgorithm::Copy(values, values_output);
      return;
    }

    //we need to determine based on the keys what is the keystate for
    //each key. The states are start, middle, end of a series and the special
    //state start and end of a series
    vtkm::cont::ArrayHandle<ReduceKeySeriesStates> keystate;

    {
      vtkm::cont::Token token;
      auto inputPortal = keys.PrepareForInput(DeviceAdapterTag(), token);
      auto keyStatePortal = keystate.PrepareForOutput(numberOfKeys, DeviceAdapterTag(), token);
      ReduceStencilGeneration<decltype(inputPortal), decltype(keyStatePortal)> kernel(
        inputPortal, keyStatePortal);
      DerivedAlgorithm::Schedule(kernel, numberOfKeys);
    }

    //next step is we need to reduce the values for each key. This is done
    //by running an inclusive scan over the values array using the stencil.
    //
    // this inclusive scan will write out two values, the first being
    // the value summed currently, the second being 0 or 1, with 1 being used
    // when this is a value of a key we need to write ( END or START_AND_END)
    {
      vtkm::cont::ArrayHandle<ValueT> reducedValues;
      vtkm::cont::ArrayHandle<ReduceKeySeriesStates> stencil;
      auto scanInput = vtkm::cont::make_ArrayHandleZip(values, keystate);
      auto scanOutput = vtkm::cont::make_ArrayHandleZip(reducedValues, stencil);

      DerivedAlgorithm::ScanInclusive(
        scanInput, scanOutput, ReduceByKeyAdd<BinaryFunctor>(binary_functor));
      //at this point we are done with keystate, so free the memory
      keystate.ReleaseResources();
      DerivedAlgorithm::Copy(reducedValues, values_output);
    }
  }

  //--------------------------------------------------------------------------
  // Sort
  template <typename T, class Storage, class BinaryCompare>
  VTKM_CONT static void Sort(vtkm::cont::ArrayHandle<T, Storage>& values,
                             BinaryCompare binary_compare)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id numValues = values.GetNumberOfValues();
    if (numValues < 2)
    {
      return;
    }
    vtkm::Id numThreads = 1;
    while (numThreads < numValues)
    {
      numThreads *= 2;
    }
    numThreads /= 2;

    vtkm::cont::Token token;

    auto portal = values.PrepareForInPlace(DeviceAdapterTag(), token);
    using MergeKernel = BitonicSortMergeKernel<decltype(portal), BinaryCompare>;
    using CrossoverKernel = BitonicSortCrossoverKernel<decltype(portal), BinaryCompare>;

    for (vtkm::Id crossoverSize = 1; crossoverSize < numValues; crossoverSize *= 2)
    {
      DerivedAlgorithm::Schedule(CrossoverKernel(portal, binary_compare, crossoverSize),
                                 numThreads);
      for (vtkm::Id mergeSize = crossoverSize / 2; mergeSize > 0; mergeSize /= 2)
      {
        DerivedAlgorithm::Schedule(MergeKernel(portal, binary_compare, mergeSize), numThreads);
      }
    }
  }

  template <typename T, class Storage>
  VTKM_CONT static void Sort(vtkm::cont::ArrayHandle<T, Storage>& values)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    DerivedAlgorithm::Sort(values, DefaultCompareFunctor());
  }

  //--------------------------------------------------------------------------
  // Sort by Key
  template <typename T, typename U, class StorageT, class StorageU>
  VTKM_CONT static void SortByKey(vtkm::cont::ArrayHandle<T, StorageT>& keys,
                                  vtkm::cont::ArrayHandle<U, StorageU>& values)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    //combine the keys and values into a ZipArrayHandle
    //we than need to specify a custom compare function wrapper
    //that only checks for key side of the pair, using a custom compare functor.
    auto zipHandle = vtkm::cont::make_ArrayHandleZip(keys, values);
    DerivedAlgorithm::Sort(zipHandle, internal::KeyCompare<T, U>());
  }

  template <typename T, typename U, class StorageT, class StorageU, class BinaryCompare>
  VTKM_CONT static void SortByKey(vtkm::cont::ArrayHandle<T, StorageT>& keys,
                                  vtkm::cont::ArrayHandle<U, StorageU>& values,
                                  BinaryCompare binary_compare)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    //combine the keys and values into a ZipArrayHandle
    //we than need to specify a custom compare function wrapper
    //that only checks for key side of the pair, using the custom compare
    //functor that the user passed in
    auto zipHandle = vtkm::cont::make_ArrayHandleZip(keys, values);
    DerivedAlgorithm::Sort(zipHandle, internal::KeyCompare<T, U, BinaryCompare>(binary_compare));
  }

  template <typename T,
            typename U,
            typename V,
            typename StorageT,
            typename StorageU,
            typename StorageV,
            typename BinaryFunctor>
  VTKM_CONT static void Transform(const vtkm::cont::ArrayHandle<T, StorageT>& input1,
                                  const vtkm::cont::ArrayHandle<U, StorageU>& input2,
                                  vtkm::cont::ArrayHandle<V, StorageV>& output,
                                  BinaryFunctor binaryFunctor)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id numValues = vtkm::Min(input1.GetNumberOfValues(), input2.GetNumberOfValues());
    if (numValues <= 0)
    {
      return;
    }

    vtkm::cont::Token token;

    auto input1Portal = input1.PrepareForInput(DeviceAdapterTag(), token);
    auto input2Portal = input2.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(numValues, DeviceAdapterTag(), token);

    BinaryTransformKernel<decltype(input1Portal),
                          decltype(input2Portal),
                          decltype(outputPortal),
                          BinaryFunctor>
      binaryKernel(input1Portal, input2Portal, outputPortal, binaryFunctor);
    DerivedAlgorithm::Schedule(binaryKernel, numValues);
  }

  //};
  //--------------------------------------------------------------------------
  // Unique
  template <typename T, class Storage>
  VTKM_CONT static void Unique(vtkm::cont::ArrayHandle<T, Storage>& values)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    DerivedAlgorithm::Unique(values, vtkm::Equal());
  }

  template <typename T, class Storage, class BinaryCompare>
  VTKM_CONT static void Unique(vtkm::cont::ArrayHandle<T, Storage>& values,
                               BinaryCompare binary_compare)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::cont::ArrayHandle<vtkm::Id, vtkm::cont::StorageTagBasic> stencilArray;
    vtkm::Id inputSize = values.GetNumberOfValues();

    using WrappedBOpType = internal::WrappedBinaryOperator<bool, BinaryCompare>;
    WrappedBOpType wrappedCompare(binary_compare);

    {
      vtkm::cont::Token token;
      auto valuesPortal = values.PrepareForInput(DeviceAdapterTag(), token);
      auto stencilPortal = stencilArray.PrepareForOutput(inputSize, DeviceAdapterTag(), token);
      ClassifyUniqueComparisonKernel<decltype(valuesPortal),
                                     decltype(stencilPortal),
                                     WrappedBOpType>
        classifyKernel(valuesPortal, stencilPortal, wrappedCompare);

      DerivedAlgorithm::Schedule(classifyKernel, inputSize);
    }

    vtkm::cont::ArrayHandle<T, vtkm::cont::StorageTagBasic> outputArray;

    DerivedAlgorithm::CopyIf(values, stencilArray, outputArray);

    values.Allocate(outputArray.GetNumberOfValues());
    DerivedAlgorithm::Copy(outputArray, values);
  }

  //--------------------------------------------------------------------------
  // Upper bounds
  template <typename T, class CIn, class CVal, class COut>
  VTKM_CONT static void UpperBounds(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                    const vtkm::cont::ArrayHandle<T, CVal>& values,
                                    vtkm::cont::ArrayHandle<vtkm::Id, COut>& output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id arraySize = values.GetNumberOfValues();

    vtkm::cont::Token token;

    auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
    auto valuesPortal = values.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(arraySize, DeviceAdapterTag(), token);

    UpperBoundsKernel<decltype(inputPortal), decltype(valuesPortal), decltype(outputPortal)> kernel(
      inputPortal, valuesPortal, outputPortal);
    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template <typename T, class CIn, class CVal, class COut, class BinaryCompare>
  VTKM_CONT static void UpperBounds(const vtkm::cont::ArrayHandle<T, CIn>& input,
                                    const vtkm::cont::ArrayHandle<T, CVal>& values,
                                    vtkm::cont::ArrayHandle<vtkm::Id, COut>& output,
                                    BinaryCompare binary_compare)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    vtkm::Id arraySize = values.GetNumberOfValues();

    vtkm::cont::Token token;

    auto inputPortal = input.PrepareForInput(DeviceAdapterTag(), token);
    auto valuesPortal = values.PrepareForInput(DeviceAdapterTag(), token);
    auto outputPortal = output.PrepareForOutput(arraySize, DeviceAdapterTag(), token);

    UpperBoundsKernelComparisonKernel<decltype(inputPortal),
                                      decltype(valuesPortal),
                                      decltype(outputPortal),
                                      BinaryCompare>
      kernel(inputPortal, valuesPortal, outputPortal, binary_compare);

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template <class CIn, class COut>
  VTKM_CONT static void UpperBounds(const vtkm::cont::ArrayHandle<vtkm::Id, CIn>& input,
                                    vtkm::cont::ArrayHandle<vtkm::Id, COut>& values_output)
  {
    VTKM_LOG_SCOPE_FUNCTION(vtkm::cont::LogLevel::Perf);

    DeviceAdapterAlgorithmGeneral<DerivedAlgorithm, DeviceAdapterTag>::UpperBounds(
      input, values_output, values_output);
  }
};

} // namespace internal

/// \brief Class providing a device-specific support for selecting the optimal
/// Task type for a given worklet.
///
/// When worklets are launched inside the execution environment we need to
/// ask the device adapter what is the preferred execution style, be it
/// a tiled iteration pattern, or strided. This class
///
/// By default if not specialized for a device adapter the default
/// is to use vtkm::exec::internal::TaskSingular
///
template <typename DeviceTag>
class DeviceTaskTypes
{
public:
  template <typename WorkletType, typename InvocationType>
  static vtkm::exec::internal::TaskSingular<WorkletType, InvocationType> MakeTask(
    WorkletType& worklet,
    InvocationType& invocation,
    vtkm::Id,
    vtkm::Id globalIndexOffset = 0)
  {
    using Task = vtkm::exec::internal::TaskSingular<WorkletType, InvocationType>;
    return Task(worklet, invocation, globalIndexOffset);
  }

  template <typename WorkletType, typename InvocationType>
  static vtkm::exec::internal::TaskSingular<WorkletType, InvocationType> MakeTask(
    WorkletType& worklet,
    InvocationType& invocation,
    vtkm::Id3,
    vtkm::Id globalIndexOffset = 0)
  {
    using Task = vtkm::exec::internal::TaskSingular<WorkletType, InvocationType>;
    return Task(worklet, invocation, globalIndexOffset);
  }
};
}
} // namespace vtkm::cont

#endif //vtk_m_cont_internal_DeviceAdapterAlgorithmGeneral_h
