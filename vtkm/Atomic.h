//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_Atomic_h
#define vtk_m_Atomic_h

#include <vtkm/List.h>

#include <vtkm/internal/Windows.h>

#include <atomic>

namespace vtkm
{

/// \brief Specifies memory order semantics for atomic operations.
///
/// The memory order parameter controls how all other memory operations are
/// ordered around a specific atomic instruction.
///
/// Memory access is complicated. Compilers can reorder instructions to optimize
/// scheduling, processors can speculatively read memory, and caches make
/// assumptions about coherency that we may not normally be aware of. Because of
/// this complexity, the order in which multiple updates to shared memory become
/// visible to other threads is not guaranteed, nor is it guaranteed that each
/// thread will see memory updates occur in the same order as any other thread.
/// This can lead to surprising behavior and cause problems when using atomics
/// to communicate between threads.
///
/// These problems are solved by using a standard set of memory orderings which
/// describe common access patterns used for shared memory programming. Their
/// goal is to provide guarantees that changes made in one thread will be visible
/// to another thread at a specific and predictable point in execution, regardless
/// of any hardware or compiler optimizations.
///
/// If unsure, use `SequentiallyConsistent` memory orderings. It will "do the right
/// thing", but at the cost of increased and possibly unnecessary memory ordering
/// restrictions. The other orderings are optimizations that are only applicable
/// in very specific situations.
///
/// See https://en.cppreference.com/w/cpp/atomic/memory_order for a detailed
/// description of the different orderings and their usage.
///
/// The memory order semantics follow those of other common atomic operations such as
/// the `std::memory_order` identifiers used for `std::atomic`.
///
/// Note that when a memory order is specified, the enforced memory order is guaranteed
/// to be as good or better than that requested.
///
enum class MemoryOrder
{
  /// An atomic operations with `Relaxed` memory order enforces no synchronization or ordering
  /// constraints on local reads and writes. That is, a read or write to a local, non-atomic
  /// variable may be moved to before or after an atomic operation with `Relaxed` memory order.
  ///
  Relaxed,

  /// A load operation with `Acquire` memory order will enforce that any local read or write
  /// operations listed in the program after the atomic will happen after the atomic.
  ///
  Acquire,

  /// A store operation with `Release` memory order will enforce that any local read or write
  /// operations listed in the program before the atomic will happen before the atomic.
  ///
  Release,

  /// A read-modify-write operation with `AcquireAndRelease` memory order will enforce that any
  /// local read or write operations listed in the program before the atomic will happen before the
  /// atomic and likewise any read or write operations listed in the program after the atomic will
  /// happen after the atomic.
  ///
  AcquireAndRelease,

  /// An atomic with `SequentiallyConsistent` memory order will enforce any appropriate semantics
  /// as `Acquire`, `Release`, and `AcquireAndRelease`. Additionally, `SequentiallyConsistent` will
  /// enforce a consistent ordering of atomic operations across all threads. That is, all threads
  /// observe the modifications in the same order.
  ///
  SequentiallyConsistent
};

namespace internal
{

VTKM_EXEC_CONT inline std::memory_order StdAtomicMemOrder(vtkm::MemoryOrder order)
{
  switch (order)
  {
    case vtkm::MemoryOrder::Relaxed:
      return std::memory_order_relaxed;
    case vtkm::MemoryOrder::Acquire:
      return std::memory_order_acquire;
    case vtkm::MemoryOrder::Release:
      return std::memory_order_release;
    case vtkm::MemoryOrder::AcquireAndRelease:
      return std::memory_order_acq_rel;
    case vtkm::MemoryOrder::SequentiallyConsistent:
      return std::memory_order_seq_cst;
  }

  // Should never reach here, but avoid compiler warnings
  return std::memory_order_seq_cst;
}

} // namespace internal

} // namespace vtkm


#if defined(VTKM_CUDA_DEVICE_PASS)

namespace vtkm
{
namespace detail
{

// Fence to ensure that previous non-atomic stores are visible to other threads.
VTKM_EXEC_CONT inline void AtomicStoreFence(vtkm::MemoryOrder order)
{
  if ((order == vtkm::MemoryOrder::Release) || (order == vtkm::MemoryOrder::AcquireAndRelease) ||
      (order == vtkm::MemoryOrder::SequentiallyConsistent))
  {
    __threadfence();
  }
}

// Fence to ensure that previous non-atomic stores are visible to other threads.
VTKM_EXEC_CONT inline void AtomicLoadFence(vtkm::MemoryOrder order)
{
  if ((order == vtkm::MemoryOrder::Acquire) || (order == vtkm::MemoryOrder::AcquireAndRelease) ||
      (order == vtkm::MemoryOrder::SequentiallyConsistent))
  {
    __threadfence();
  }
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicLoadImpl(T* const addr, vtkm::MemoryOrder order)
{
  volatile T* const vaddr = addr; /* volatile to bypass cache*/
  if (order == vtkm::MemoryOrder::SequentiallyConsistent)
  {
    __threadfence();
  }
  const T value = *vaddr;
  /* fence to ensure that dependent reads are correctly ordered */
  AtomicLoadFence(order);
  return value;
}

template <typename T>
VTKM_EXEC_CONT inline void AtomicStoreImpl(T* addr, T value, vtkm::MemoryOrder order)
{
  volatile T* vaddr = addr; /* volatile to bypass cache */
  /* fence to ensure that previous non-atomic stores are visible to other threads */
  AtomicStoreFence(order);
  *vaddr = value;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicAddImpl(T* addr, T arg, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  auto result = atomicAdd(addr, arg);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicAndImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  auto result = atomicAnd(addr, mask);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicOrImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  auto result = atomicOr(addr, mask);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicXorImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  auto result = atomicXor(addr, mask);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicNotImpl(T* addr, vtkm::MemoryOrder order)
{
  return AtomicXorImpl(addr, static_cast<T>(~T{ 0u }), order);
}

template <typename T>
VTKM_EXEC_CONT inline bool AtomicCompareExchangeImpl(T* addr,
                                                     T* expected,
                                                     T desired,
                                                     vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  auto result = atomicCAS(addr, *expected, desired);
  AtomicLoadFence(order);
  if (result == *expected)
  {
    return true;
  }
  else
  {
    *expected = result;
    return false;
  }
}
#if __CUDA_ARCH__ < 200
VTKM_EXEC_CONT inline vtkm::Float32 AtomicAddImpl(vtkm::Float32* address,
                                                  vtkm::Float32 value,
                                                  vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  vtkm::UInt32 assumed;
  vtkm::UInt32 old = __float_as_int(*address);
  do
  {
    assumed = old;
    old = atomicCAS(reinterpret_cast<vtkm::UInt32*>(address),
                    assumed,
                    __float_as_int(__int_as_float(assumed) + value));
  } while (assumed != old);
  AtomicLoadFence(order);
  return __int_as_float(old);
}
#endif
#if __CUDA_ARCH__ < 600
VTKM_EXEC_CONT inline vtkm::Float64 AtomicAddImpl(vtkm::Float64* address,
                                                  vtkm::Float64 value,
                                                  vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  vtkm::UInt64 assumed;
  vtkm::UInt64 old = __double_as_longlong(*address);
  do
  {
    assumed = old;
    old = atomicCAS(reinterpret_cast<vtkm::UInt64*>(address),
                    assumed,
                    __double_as_longlong(__longlong_as_double(assumed) + value));
  } while (assumed != old);
  AtomicLoadFence(order);
  return __longlong_as_double(old);
}
#endif
}
} // namespace vtkm::detail

#elif defined(VTKM_ENABLE_KOKKOS)

VTKM_THIRDPARTY_PRE_INCLUDE
// Superhack! Kokkos_Macros.hpp defines macros to include modifiers like __device__.
// However, we don't want to actually use those if compiling this with a standard
// C++ compiler (because this particular code does not run on a device). Thus,
// we want to disable that behavior when not using the device compiler. To do that,
// we are going to have to load the KokkosCore_config.h file (which you are not
// supposed to do), then undefine the device enables if necessary, then load
// Kokkos_Macros.hpp to finish the state.
#ifndef KOKKOS_MACROS_HPP
#define KOKKOS_MACROS_HPP
#include <KokkosCore_config.h>
#undef KOKKOS_MACROS_HPP
#define KOKKOS_DONT_INCLUDE_CORE_CONFIG_H

#if defined(KOKKOS_ENABLE_CUDA) && !defined(VTKM_CUDA)
#undef KOKKOS_ENABLE_CUDA

// In later versions we need to directly deactivate Kokkos_Setup_Cuda.hpp
#if KOKKOS_VERSION >= 30401
#define KOKKOS_CUDA_SETUP_HPP_
#endif
#endif

#if defined(KOKKOS_ENABLE_HIP) && !defined(VTKM_HIP)
#undef KOKKOS_ENABLE_HIP
#endif

#endif //KOKKOS_MACROS_HPP not loaded

#include <Kokkos_Atomic.hpp>
VTKM_THIRDPARTY_POST_INCLUDE

namespace vtkm
{
namespace detail
{

// Fence to ensure that previous non-atomic stores are visible to other threads.
VTKM_EXEC_CONT inline void AtomicStoreFence(vtkm::MemoryOrder order)
{
  if ((order == vtkm::MemoryOrder::Release) || (order == vtkm::MemoryOrder::AcquireAndRelease) ||
      (order == vtkm::MemoryOrder::SequentiallyConsistent))
  {
    Kokkos::memory_fence();
  }
}

// Fence to ensure that previous non-atomic stores are visible to other threads.
VTKM_EXEC_CONT inline void AtomicLoadFence(vtkm::MemoryOrder order)
{
  if ((order == vtkm::MemoryOrder::Acquire) || (order == vtkm::MemoryOrder::AcquireAndRelease) ||
      (order == vtkm::MemoryOrder::SequentiallyConsistent))
  {
    Kokkos::memory_fence();
  }
}
#ifdef KOKKOS_INTERNAL_NOT_PARALLEL
#define VTKM_DESUL_MEM_SCOPE desul::MemoryScopeCaller()
#else
#define VTKM_DESUL_MEM_SCOPE desul::MemoryScopeDevice()
#endif

template <typename T>
VTKM_EXEC_CONT inline T AtomicLoadImpl(T* const addr, vtkm::MemoryOrder order)
{
  switch (order)
  {
    case vtkm::MemoryOrder::Relaxed:
      return desul::atomic_load(addr, desul::MemoryOrderRelaxed(), VTKM_DESUL_MEM_SCOPE);
    case vtkm::MemoryOrder::Acquire:
    case vtkm::MemoryOrder::Release:           // Release doesn't make sense. Use Acquire.
    case vtkm::MemoryOrder::AcquireAndRelease: // Release doesn't make sense. Use Acquire.
      return desul::atomic_load(addr, desul::MemoryOrderAcquire(), VTKM_DESUL_MEM_SCOPE);
    case vtkm::MemoryOrder::SequentiallyConsistent:
      return desul::atomic_load(addr, desul::MemoryOrderSeqCst(), VTKM_DESUL_MEM_SCOPE);
  }

  // Should never reach here, but avoid compiler warnings
  return desul::atomic_load(addr, desul::MemoryOrderSeqCst(), VTKM_DESUL_MEM_SCOPE);
}

template <typename T>
VTKM_EXEC_CONT inline void AtomicStoreImpl(T* addr, T value, vtkm::MemoryOrder order)
{
  switch (order)
  {
    case vtkm::MemoryOrder::Relaxed:
      desul::atomic_store(addr, value, desul::MemoryOrderRelaxed(), VTKM_DESUL_MEM_SCOPE);
      break;
    case vtkm::MemoryOrder::Acquire: // Acquire doesn't make sense. Use Release.
    case vtkm::MemoryOrder::Release:
    case vtkm::MemoryOrder::AcquireAndRelease: // Acquire doesn't make sense. Use Release.
      desul::atomic_store(addr, value, desul::MemoryOrderRelease(), VTKM_DESUL_MEM_SCOPE);
      break;
    case vtkm::MemoryOrder::SequentiallyConsistent:
      desul::atomic_store(addr, value, desul::MemoryOrderSeqCst(), VTKM_DESUL_MEM_SCOPE);
      break;
  }
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicAddImpl(T* addr, T arg, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  T result = Kokkos::atomic_fetch_add(addr, arg);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicAndImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  T result = Kokkos::atomic_fetch_and(addr, mask);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicOrImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  T result = Kokkos::atomic_fetch_or(addr, mask);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicXorImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  T result = Kokkos::atomic_fetch_xor(addr, mask);
  AtomicLoadFence(order);
  return result;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicNotImpl(T* addr, vtkm::MemoryOrder order)
{
  return AtomicXorImpl(addr, static_cast<T>(~T{ 0u }), order);
}

template <typename T>
VTKM_EXEC_CONT inline bool AtomicCompareExchangeImpl(T* addr,
                                                     T* expected,
                                                     T desired,
                                                     vtkm::MemoryOrder order)
{
  AtomicStoreFence(order);
  T oldValue = Kokkos::atomic_compare_exchange(addr, *expected, desired);
  AtomicLoadFence(order);
  if (oldValue == *expected)
  {
    return true;
  }
  else
  {
    *expected = oldValue;
    return false;
  }
}
}
} // namespace vtkm::detail

#elif defined(VTKM_MSVC)

// Supports vtkm::UInt8, vtkm::UInt16, vtkm::UInt32, vtkm::UInt64

#include <cstdint>
#include <cstring>
#include <intrin.h> // For MSVC atomics

namespace vtkm
{
namespace detail
{

template <typename To, typename From>
VTKM_EXEC_CONT inline To BitCast(const From& src)
{
  // The memcpy should be removed by the compiler when possible, but this
  // works around a host of issues with bitcasting using reinterpret_cast.
  VTKM_STATIC_ASSERT(sizeof(From) == sizeof(To));
  To dst;
  std::memcpy(&dst, &src, sizeof(From));
  return dst;
}

template <typename T>
VTKM_EXEC_CONT inline T BitCast(T&& src)
{
  return std::forward<T>(src);
}

// Note about Load and Store implementations:
//
// "Simple reads and writes to properly-aligned 32-bit variables are atomic
//  operations"
//
// "Simple reads and writes to properly aligned 64-bit variables are atomic on
// 64-bit Windows. Reads and writes to 64-bit values are not guaranteed to be
// atomic on 32-bit Windows."
//
// "Reads and writes to variables of other sizes [than 32 or 64 bits] are not
// guaranteed to be atomic on any platform."
//
// https://docs.microsoft.com/en-us/windows/desktop/sync/interlocked-variable-access

VTKM_EXEC_CONT inline vtkm::UInt8 AtomicLoadImpl(vtkm::UInt8* const addr, vtkm::MemoryOrder order)
{
  // This assumes that the memory interface is smart enough to load a 32-bit
  // word atomically and a properly aligned 8-bit word from it.
  // We could build address masks and do shifts to perform this manually if
  // this assumption is incorrect.
  auto result = *static_cast<volatile vtkm::UInt8* const>(addr);
  std::atomic_thread_fence(internal::StdAtomicMemOrder(order));
  return result;
}
VTKM_EXEC_CONT inline vtkm::UInt16 AtomicLoadImpl(vtkm::UInt16* const addr, vtkm::MemoryOrder order)
{
  // This assumes that the memory interface is smart enough to load a 32-bit
  // word atomically and a properly aligned 16-bit word from it.
  // We could build address masks and do shifts to perform this manually if
  // this assumption is incorrect.
  auto result = *static_cast<volatile vtkm::UInt16* const>(addr);
  std::atomic_thread_fence(internal::StdAtomicMemOrder(order));
  return result;
}
VTKM_EXEC_CONT inline vtkm::UInt32 AtomicLoadImpl(vtkm::UInt32* const addr, vtkm::MemoryOrder order)
{
  auto result = *static_cast<volatile vtkm::UInt32* const>(addr);
  std::atomic_thread_fence(internal::StdAtomicMemOrder(order));
  return result;
}
VTKM_EXEC_CONT inline vtkm::UInt64 AtomicLoadImpl(vtkm::UInt64* const addr, vtkm::MemoryOrder order)
{
  auto result = *static_cast<volatile vtkm::UInt64* const>(addr);
  std::atomic_thread_fence(internal::StdAtomicMemOrder(order));
  return result;
}

VTKM_EXEC_CONT inline void AtomicStoreImpl(vtkm::UInt8* addr,
                                           vtkm::UInt8 val,
                                           vtkm::MemoryOrder vtkmNotUsed(order))
{
  // There doesn't seem to be an atomic store instruction in the windows
  // API, so just exchange and discard the result.
  _InterlockedExchange8(reinterpret_cast<volatile CHAR*>(addr), BitCast<CHAR>(val));
}
VTKM_EXEC_CONT inline void AtomicStoreImpl(vtkm::UInt16* addr,
                                           vtkm::UInt16 val,
                                           vtkm::MemoryOrder vtkmNotUsed(order))
{
  // There doesn't seem to be an atomic store instruction in the windows
  // API, so just exchange and discard the result.
  _InterlockedExchange16(reinterpret_cast<volatile SHORT*>(addr), BitCast<SHORT>(val));
}
VTKM_EXEC_CONT inline void AtomicStoreImpl(vtkm::UInt32* addr,
                                           vtkm::UInt32 val,
                                           vtkm::MemoryOrder order)
{
  std::atomic_thread_fence(internal::StdAtomicMemOrder(order));
  *addr = val;
}
VTKM_EXEC_CONT inline void AtomicStoreImpl(vtkm::UInt64* addr,
                                           vtkm::UInt64 val,
                                           vtkm::MemoryOrder order)
{
  std::atomic_thread_fence(internal::StdAtomicMemOrder(order));
  *addr = val;
}

#define VTKM_ATOMIC_OP(vtkmName, winName, vtkmType, winType, suffix)                             \
  VTKM_EXEC_CONT inline vtkmType vtkmName(vtkmType* addr, vtkmType arg, vtkm::MemoryOrder order) \
  {                                                                                              \
    return BitCast<vtkmType>(                                                                    \
      winName##suffix(reinterpret_cast<volatile winType*>(addr), BitCast<winType>(arg)));        \
  }

#define VTKM_ATOMIC_OPS_FOR_TYPE(vtkmType, winType, suffix)                                     \
  VTKM_ATOMIC_OP(AtomicAddImpl, _InterlockedExchangeAdd, vtkmType, winType, suffix)             \
  VTKM_ATOMIC_OP(AtomicAndImpl, _InterlockedAnd, vtkmType, winType, suffix)                     \
  VTKM_ATOMIC_OP(AtomicOrImpl, _InterlockedOr, vtkmType, winType, suffix)                       \
  VTKM_ATOMIC_OP(AtomicXorImpl, _InterlockedXor, vtkmType, winType, suffix)                     \
  VTKM_EXEC_CONT inline vtkmType AtomicNotImpl(vtkmType* addr, vtkm::MemoryOrder order)         \
  {                                                                                             \
    return AtomicXorImpl(addr, static_cast<vtkmType>(~vtkmType{ 0u }), order);                  \
  }                                                                                             \
  VTKM_EXEC_CONT inline bool AtomicCompareExchangeImpl(                                         \
    vtkmType* addr, vtkmType* expected, vtkmType desired, vtkm::MemoryOrder vtkmNotUsed(order)) \
  {                                                                                             \
    vtkmType result = BitCast<vtkmType>(                                                        \
      _InterlockedCompareExchange##suffix(reinterpret_cast<volatile winType*>(addr),            \
                                          BitCast<winType>(desired),                            \
                                          BitCast<winType>(*expected)));                        \
    if (result == *expected)                                                                    \
    {                                                                                           \
      return true;                                                                              \
    }                                                                                           \
    else                                                                                        \
    {                                                                                           \
      *expected = result;                                                                       \
      return false;                                                                             \
    }                                                                                           \
  }

VTKM_ATOMIC_OPS_FOR_TYPE(vtkm::UInt8, CHAR, 8)
VTKM_ATOMIC_OPS_FOR_TYPE(vtkm::UInt16, SHORT, 16)
VTKM_ATOMIC_OPS_FOR_TYPE(vtkm::UInt32, LONG, )
VTKM_ATOMIC_OPS_FOR_TYPE(vtkm::UInt64, LONG64, 64)

#undef VTKM_ATOMIC_OPS_FOR_TYPE

VTKM_EXEC_CONT inline vtkm::Float32 AtomicAddImpl(vtkm::Float32* address,
                                                  vtkm::Float32 value,
                                                  vtkm::MemoryOrder vtkmNotUsed(order))
{
  LONG assumed;
  LONG old = BitCast<LONG>(*address);
  do
  {
    assumed = old;
    old = _InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(address),
                                      BitCast<LONG>(BitCast<vtkm::Float32>(assumed) + value),
                                      assumed);
  } while (assumed != old);
  return BitCast<vtkm::Float32>(old);
}

VTKM_EXEC_CONT inline vtkm::Float64 AtomicAddImpl(vtkm::Float64* address,
                                                  vtkm::Float64 value,
                                                  vtkm::MemoryOrder vtkmNotUsed(order))
{
  LONG64 assumed;
  LONG64 old = BitCast<LONG64>(*address);
  do
  {
    assumed = old;
    old = _InterlockedCompareExchange64(reinterpret_cast<volatile LONG64*>(address),
                                        BitCast<LONG64>(BitCast<vtkm::Float64>(assumed) + value),
                                        assumed);
  } while (assumed != old);
  return BitCast<vtkm::Float64>(old);
}

}
} // namespace vtkm::detail

#else // gcc/clang for CPU

// Supports vtkm::UInt8, vtkm::UInt16, vtkm::UInt32, vtkm::UInt64

#include <cstdint>
#include <cstring>

namespace vtkm
{
namespace detail
{

VTKM_EXEC_CONT inline int GccAtomicMemOrder(vtkm::MemoryOrder order)
{
  switch (order)
  {
    case vtkm::MemoryOrder::Relaxed:
      return __ATOMIC_RELAXED;
    case vtkm::MemoryOrder::Acquire:
      return __ATOMIC_ACQUIRE;
    case vtkm::MemoryOrder::Release:
      return __ATOMIC_RELEASE;
    case vtkm::MemoryOrder::AcquireAndRelease:
      return __ATOMIC_ACQ_REL;
    case vtkm::MemoryOrder::SequentiallyConsistent:
      return __ATOMIC_SEQ_CST;
  }

  // Should never reach here, but avoid compiler warnings
  return __ATOMIC_SEQ_CST;
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicLoadImpl(T* const addr, vtkm::MemoryOrder order)
{
  return __atomic_load_n(addr, GccAtomicMemOrder(order));
}

template <typename T>
VTKM_EXEC_CONT inline void AtomicStoreImpl(T* addr, T value, vtkm::MemoryOrder order)
{
  return __atomic_store_n(addr, value, GccAtomicMemOrder(order));
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicAddImpl(T* addr, T arg, vtkm::MemoryOrder order)
{
  return __atomic_fetch_add(addr, arg, GccAtomicMemOrder(order));
}

#include <vtkmstd/bit_cast.h>

// TODO: Use enable_if to write one version for both Float32 and Float64.
VTKM_EXEC_CONT inline vtkm::Float32 AtomicAddImpl(vtkm::Float32* addr,
                                                  vtkm::Float32 arg,
                                                  vtkm::MemoryOrder order)
{
  vtkm::UInt32 expected = vtkmstd::bit_cast<vtkm::UInt32>(*addr);
  vtkm::UInt32 desired;

  do
  {
    desired = vtkmstd::bit_cast<vtkm::UInt32>(vtkmstd::bit_cast<vtkm::Float32>(expected) + arg);
  } while (
    !__atomic_compare_exchange_n(reinterpret_cast<vtkm::UInt32*>(addr),
                                 &expected, // reloads expected with *addr prior to the operation
                                 desired,
                                 false,
                                 GccAtomicMemOrder(order),
                                 GccAtomicMemOrder(order)));
  // return the "old" value that was in the memory.
  return vtkmstd::bit_cast<vtkm::Float32>(expected);
}

// TODO: Use enable_if to write one version for both Float32 and Float64.
VTKM_EXEC_CONT inline vtkm::Float64 AtomicAddImpl(vtkm::Float64* addr,
                                                  vtkm::Float64 arg,
                                                  vtkm::MemoryOrder order)
{
  vtkm::UInt64 expected = vtkmstd::bit_cast<vtkm::UInt64>(*addr);
  vtkm::UInt64 desired;

  do
  {
    desired = vtkmstd::bit_cast<vtkm::UInt64>(vtkmstd::bit_cast<vtkm::Float64>(expected) + arg);
  } while (
    !__atomic_compare_exchange_n(reinterpret_cast<vtkm::UInt64*>(addr),
                                 &expected, // reloads expected with *addr prior to the operation
                                 desired,
                                 false,
                                 GccAtomicMemOrder(order),
                                 GccAtomicMemOrder(order)));
  // return the "old" value that was in the memory.
  return vtkmstd::bit_cast<vtkm::Float64>(expected);
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicAndImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  return __atomic_fetch_and(addr, mask, GccAtomicMemOrder(order));
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicOrImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  return __atomic_fetch_or(addr, mask, GccAtomicMemOrder(order));
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicXorImpl(T* addr, T mask, vtkm::MemoryOrder order)
{
  return __atomic_fetch_xor(addr, mask, GccAtomicMemOrder(order));
}

template <typename T>
VTKM_EXEC_CONT inline T AtomicNotImpl(T* addr, vtkm::MemoryOrder order)
{
  return AtomicXorImpl(addr, static_cast<T>(~T{ 0u }), order);
}

template <typename T>
VTKM_EXEC_CONT inline bool AtomicCompareExchangeImpl(T* addr,
                                                     T* expected,
                                                     T desired,
                                                     vtkm::MemoryOrder order)
{
  return __atomic_compare_exchange_n(
    addr, expected, desired, false, GccAtomicMemOrder(order), GccAtomicMemOrder(order));
}
}
} // namespace vtkm::detail

#endif // gcc/clang

namespace vtkm
{

namespace detail
{

template <typename T>
using OppositeSign = typename std::conditional<std::is_signed<T>::value,
                                               typename std::make_unsigned<T>::type,
                                               typename std::make_signed<T>::type>::type;

} // namespace detail

/// \brief The preferred type to use for atomic operations.
///
using AtomicTypePreferred = vtkm::UInt32;

/// \brief A list of types that can be used with atomic operations.
///
/// TODO: Adjust based on devices being compiled.
///
/// BUG: vtkm::UInt64 is provided in this list even though it is not supported on CUDA
/// before compute capability 3.5.
///
using AtomicTypesSupported = vtkm::List<vtkm::UInt32, vtkm::UInt64>;

/// \brief Atomic function to load a value from a shared memory location.
///
/// Given a pointer, returns the value in that pointer. If other threads are writing to
/// that same location, the returned value will be consistent to what was present before
/// or after that write.
///
template <typename T>
VTKM_EXEC_CONT inline T AtomicLoad(T* const pointer,
                                   vtkm::MemoryOrder order = vtkm::MemoryOrder::Acquire)
{
  return detail::AtomicLoadImpl(pointer, order);
}

///@{
/// \brief Atomic function to save a value to a shared memory location.
///
/// Given a pointer and a value, stores that value at the pointer's location. If two
/// threads are simultaneously using `AtomicStore` at the same location, the resulting
/// value will be one of the values or the other (as opposed to a mix of bits).
///
template <typename T>
VTKM_EXEC_CONT inline void AtomicStore(T* pointer,
                                       T value,
                                       vtkm::MemoryOrder order = vtkm::MemoryOrder::Release)
{
  detail::AtomicStoreImpl(pointer, value, order);
}
template <typename T>
VTKM_EXEC_CONT inline void AtomicStore(T* pointer,
                                       detail::OppositeSign<T> value,
                                       vtkm::MemoryOrder order = vtkm::MemoryOrder::Release)
{
  detail::AtomicStoreImpl(pointer, static_cast<T>(value), order);
}
///@}

///@{
/// \brief Atomic function to add a value to a shared memory location.
///
/// Given a pointer and an operand, adds the operand to the value at the given memory
/// location. The result of the addition is put into that memory location and the
/// _old_ value that was originally in the memory is returned. For example, if you
/// call `AtomicAdd` on a memory location that holds a 5 with an operand of 3, the
/// value of 8 is stored in the memory location and the value of 5 is returned.
///
/// If multiple threads call `AtomicAdd` simultaneously, they will not interfere with
/// each other. The result will be consistent as if one was called before the other
/// (although it is indeterminate which will be applied first).
///
template <typename T>
VTKM_EXEC_CONT inline T AtomicAdd(
  T* pointer,
  T operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicAddImpl(pointer, operand, order);
}
template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
VTKM_EXEC_CONT inline T AtomicAdd(
  T* pointer,
  detail::OppositeSign<T> operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicAddImpl(pointer, static_cast<T>(operand), order);
}
///@}

///@{
/// \brief Atomic function to AND bits to a shared memory location.
///
/// Given a pointer and an operand, performs a bitwise AND of the operand and thevalue at the given
/// memory location. The result of the AND is put into that memory location and the _old_ value
/// that was originally in the memory is returned. For example, if you call `AtomicAnd` on a memory
/// location that holds a 0x6 with an operand of 0x3, the value of 0x2 is stored in the memory
/// location and the value of 0x6 is returned.
///
/// If multiple threads call `AtomicAnd` simultaneously, they will not interfere with
/// each other. The result will be consistent as if one was called before the other
/// (although it is indeterminate which will be applied first).
///
template <typename T>
VTKM_EXEC_CONT inline T AtomicAnd(
  T* pointer,
  T operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicAndImpl(pointer, operand, order);
}
template <typename T>
VTKM_EXEC_CONT inline T AtomicAnd(
  T* pointer,
  detail::OppositeSign<T> operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicAndImpl(pointer, static_cast<T>(operand), order);
}
///@}

///@{
/// \brief Atomic function to OR bits to a shared memory location.
///
/// Given a pointer and an operand, performs a bitwise OR of the operand and the value at the given
/// memory location. The result of the OR is put into that memory location and the _old_ value
/// that was originally in the memory is returned. For example, if you call `AtomicOr` on a memory
/// location that holds a 0x6 with an operand of 0x3, the value of 0x7 is stored in the memory
/// location and the value of 0x6 is returned.
///
/// If multiple threads call `AtomicOr` simultaneously, they will not interfere with
/// each other. The result will be consistent as if one was called before the other
/// (although it is indeterminate which will be applied first).
///
template <typename T>
VTKM_EXEC_CONT inline T
AtomicOr(T* pointer, T operand, vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicOrImpl(pointer, operand, order);
}
template <typename T>
VTKM_EXEC_CONT inline T AtomicOr(
  T* pointer,
  detail::OppositeSign<T> operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicOrImpl(pointer, static_cast<T>(operand), order);
}
///@}

///@{
/// \brief Atomic function to XOR bits to a shared memory location.
///
/// Given a pointer and an operand, performs a bitwise exclusive-OR of the operand and the value at
/// the given memory location. The result of the XOR is put into that memory location and the _old_
/// value that was originally in the memory is returned. For example, if you call `AtomicXor` on a
/// memory location that holds a 0x6 with an operand of 0x3, the value of 0x5 is stored in the
/// memory location and the value of 0x6 is returned.
///
/// If multiple threads call `AtomicXor` simultaneously, they will not interfere with
/// each other. The result will be consistent as if one was called before the other.
///
template <typename T>
VTKM_EXEC_CONT inline T AtomicXor(
  T* pointer,
  T operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicXorImpl(pointer, operand, order);
}
template <typename T>
VTKM_EXEC_CONT inline T AtomicXor(
  T* pointer,
  detail::OppositeSign<T> operand,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicXorImpl(pointer, static_cast<T>(operand), order);
}
///@}

/// \brief Atomic function to NOT bits to a shared memory location.
///
/// Given a pointer, performs a bitwise NOT of the value at the given
/// memory location. The result of the NOT is put into that memory location and the _old_ value
/// that was originally in the memory is returned.
///
/// If multiple threads call `AtomicNot` simultaneously, they will not interfere with
/// each other. The result will be consistent as if one was called before the other.
///
template <typename T>
VTKM_EXEC_CONT inline T AtomicNot(
  T* pointer,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicNotImpl(pointer, order);
}

/// \brief Atomic function that replaces a value given a condition.
///
/// Given a pointer to a `shared` value, a pointer holding the `expected` value at that shared
/// location, and a new `desired` value, `AtomicCompareExchange` compares the existing `shared`
/// value to the `expected` value, and then conditionally replaces the `shared` value with
/// the provided `desired` value. Otherwise, the `expected` value gets replaced with the
/// `shared` value. Note that in either case, the function returns with `expected` replaced
/// with the value _originally_ in `shared` at the start of the call.
///
/// If the `shared` value and `expected` value are the same, then `shared` gets set to
/// `desired`, and `AtomicCompareAndExchange` returns `true`.
///
/// If the `shared` value and `expected` value are different, then `expected` gets set
/// to `shared`, and `AtomicCompareAndExchange` returns `false`. The value at `shared`
/// is _not_ changed in this case.
///
/// If multiple threads call `AtomicCompareExchange` simultaneously with the same `shared`
/// pointer, the result will be consistent as if one was called before the other (although
/// it is indeterminate which will be applied first). Note that the `expected` pointer should
/// _not_ be shared among threads. The `expected` pointer should be thread-local (often
/// pointing to an object on the stack).
///
template <typename T>
VTKM_EXEC_CONT inline bool AtomicCompareExchange(
  T* shared,
  T* expected,
  T desired,
  vtkm::MemoryOrder order = vtkm::MemoryOrder::SequentiallyConsistent)
{
  return detail::AtomicCompareExchangeImpl(shared, expected, desired, order);
}

} // namespace vtkm

#endif //vtk_m_Atomic_h
