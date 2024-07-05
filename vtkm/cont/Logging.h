//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtk_m_cont_Logging_h
#define vtk_m_cont_Logging_h

#include <vtkm/internal/Configure.h>
#include <vtkm/internal/ExportMacros.h>

#include <vtkm/Types.h>

#include <vtkm/cont/vtkm_cont_export.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>

/// \file Logging.h
/// \brief Logging utilities.
///
/// This file includes the logging system for VTK-m. There are a variety of
/// macros to print log messages using C++ stream or printf syntax. Nested
/// scopes may be created in the log output, and there are several helper
/// functions to help format common types of log data such as byte counts and
/// type names.
///
/// Logging is enabled via the CMake option VTKm_ENABLE_LOGGING by default.
/// The default log level is set to only log Warn and Error messages; Fatal
/// levels are printed to stderr by default. The logging system will need
/// to be initialized through a call to either vtkm::cont::Initialize or
/// vtkm::cont::InitLogging.
///
/// Additional logging features are enabled by calling vtkm::cont::InitLogging
/// (or preferably, vtkm::cont::Initialize) in an executable. This will:
/// - Set human-readable names for the log levels in the output.
/// - Allow the stderr logging level to be set at runtime by passing a
///   '--vtkm-log-level [level]' argument to the executable.
/// - Name the main thread.
/// - Print a preamble with details of the program's startup (args, etc).
/// - Install signal handlers to automatically print stacktraces and error
///   contexts (linux only) on crashes.
///
/// The main logging entry points are the macros VTKM_LOG_S and VTKM_LOG_F,
/// which using C++ stream and printf syntax, repectively. Other variants exist,
/// including conditional logging and special-purpose logs for writing specific
/// events, such as DynamicObject cast results and TryExecute failures.
///
/// The logging backend supports the concept of "Scopes". By creating a new
/// scope with the macros VTKM_LOG_SCOPE or VTKM_LOG_SCOPE_FUNCTION, a new
/// "logging scope" is opened within the C++ scope the macro is called from. New
/// messages will be indented in the log until the scope ends, at which point
/// a message is logged with the elapsed time that the scope was active. Scopes
/// may be nested to arbitrary depths.
///
/// The logging implementation is thread-safe. When working in a multithreaded
/// environment, each thread may be assigned a human-readable name using
/// vtkm::cont::SetThreadName. This will appear in the log output so that
/// per-thread messages can be easily tracked.
///
/// By default, only Warn, Error, and Fatal messages are printed to
/// stderr. This can be changed at runtime by passing the '--vtkm-log-level' flag to an
/// executable that calls vtkm::cont::InitLogging. Alternatively, the
/// application can explicitly call vtkm::cont::SetStderrLogLevel to change the
/// verbosity. When specifying a verbosity, all log levels with enum values
/// less-than-or-equal-to the requested level are printed.
/// vtkm::cont::LogLevel::Off (or "--vtkm-log-level Off") may be used to silence the log
/// completely.
///
/// The helper functions vtkm::cont::GetHumanReadableSize and
/// vtkm::cont::GetSizeString assist in formating byte sizes to a more readable
/// format. Similarly, the vtkm::cont::TypeToString template functions provide RTTI
/// based type-name information. When logging is enabled, these use the logging
/// backend to demangle symbol names on supported platforms.
///
/// The more verbose VTK-m log levels are:
/// - Perf: Logs performance information, using the scopes feature to track
///   execution time of filters, worklets, and device algorithms with
///   microsecond resolution.
/// - MemCont / MemExec: These levels log memory allocations in the control and
///   execution environments, respectively.
/// - MemTransfer: This level logs memory transfers between the control and host
///   environments.
/// - KernelLaunches: This level logs details about each device side kernel launch
///   such as the CUDA PTX, Warps, and Grids used.
/// - Cast: Logs details of dynamic object resolution.
///
/// The log may be shared and extended by applications that use VTK-m. There
/// are two log level ranges left available for applications: User and
/// UserVerbose. The User levels may be enabled without showing any of the
/// verbose VTK-m levels, while UserVerbose levels will also enable all VTK-m
/// levels.

/// \def VTKM_LOG_S(level, ...)
/// \brief Writes a message using stream syntax to the indicated log \a level.
///
/// The ellipsis may be replaced with the log message as if constructing a C++
/// stream, e.g:
///
/// \code
/// VTKM_LOG_S(vtkm::cont::LogLevel::Perf,
///            "Executed functor " << vtkm::cont::TypeToString(functor)
///             << " on device " << deviceId.GetName());
/// \endcode

/// \def VTKM_LOG_F(level, ...)
/// \brief Writes a message using printf syntax to the indicated log \a level.
///
/// The ellipsis may be replaced with the log message as if constructing a
/// printf call, e.g:
///
/// \code
/// VTKM_LOG_F(vtkm::cont::LogLevel::Perf,
///            "Executed functor %s on device %s",
///            vtkm::cont::TypeToString(functor).c_str(),
///            deviceId.GetName().c_str());
/// \endcode

/// \def VTKM_LOG_IF_S(level, cond, ...)
/// Same as VTKM_LOG_S, but only logs if \a cond is true.

/// \def VTKM_LOG_IF_F(level, cond, ...)
/// Same as VTKM_LOG_F, but only logs if \a cond is true.

/// \def VTKM_LOG_SCOPE(level, ...)
/// Creates a new scope at the requested \a level. The log scope ends when the
/// code scope ends. The ellipses form the scope name using printf syntax.
///
/// \code
/// {
///   VTKM_LOG_SCOPE(vtkm::cont::LogLevel::Perf,
///                  "Executing filter %s",
///                  vtkm::cont::TypeToString(myFilter).c_str());
///   myFilter.Execute();
/// }
/// \endcode

/// \def VTKM_LOG_SCOPE_FUNCTION(level)
/// Equivalent to `VTKM_LOG_SCOPE(level, __func__)`

/// \def VTKM_LOG_ALWAYS_S(level, ...)
/// This ostream-style log message is always emitted, even when logging is
/// disabled at compile time.

/// \def VTKM_LOG_CAST_SUCC(inObj, outObj)
/// \brief Convenience macro for logging the successful cast of dynamic object.
/// \param inObj The dynamic object.
/// \param outObj The resulting downcasted object.

/// \def VTKM_LOG_CAST_FAIL(inObj, outType)
/// \brief Convenience macro for logging a failed cast of dynamic object.
/// \param inObj The dynamic object.
/// \param outType The candidate type (or typelist) that was unsuccessful.

/// \def VTKM_LOG_TRYEXECUTE_FAIL(errorMessage, functorName, deviceId)
/// \brief Convenience macro for logging a TryExecute failure to the Error level.
/// If logging is disabled, a message is still printed to stderr.
/// \param errorMessage The error message detailing the failure.
/// \param functorName The name of the functor (see vtkm::cont::TypeToString)
/// \param deviceId The device tag / id for the device on which the functor
/// failed.

/// \def VTKM_LOG_TRYEXECUTE_DISABLE(errorMessage, functorName, deviceId)
/// \brief Similar to VTKM_LOG_TRYEXECUTE_FAIL, but also informs the user
/// that the device has been disable for future TryExecute calls.
/// \param errorMessage The error message detailing the failure.
/// \param functorName The name of the functor (see vtkm::cont::TypeToString)
/// \param deviceId The device tag / id for the device on which the functor
/// failed.

/// \def VTKM_DEFINE_USER_LOG_LEVEL(name, offset)
/// \brief Convenience macro for creating a custom log level that is usable
/// in the other macros.  If logging is disabled this macro does nothing.
/// \param name The name to give the new log level
/// \param offset The offset from the vtkm::cont::LogLevel::UserFirst value
/// from the LogLevel enum.  Additionally moduloed against the
/// vtkm::cont::LogLevel::UserLast value
/// \note This macro is to be used for quickly setting log levels.  For a
/// more maintainable solution it is recommended to create a custom enum class
/// and then cast appropriately, as described here:
/// https://gitlab.kitware.com/vtk/vtk-m/-/issues/358#note_550157

#define VTKM_CONCAT_IMPL(s1, s2) s1##s2
#define VTKM_CONCAT(s1, s2) VTKM_CONCAT_IMPL(s1, s2)

#ifdef __COUNTER__
#define VTKM_ANONYMOUS_VARIABLE VTKM_CONCAT(vtk_m_anonymous_, __COUNTER__)
#else
#define VTKM_ANONYMOUS_VARIABLE VTKM_CONCAT(vtk_m_anonymous_, __LINE__)
#endif

#if defined(VTKM_ENABLE_LOGGING)

#define VTKM_LOG_IF_S(level, cond, ...) \
  vtkm::cont::LogCondStream(level, cond, __FILE__, __LINE__) << __VA_ARGS__

#define VTKM_LOG_IF_F(level, cond, ...) \
  vtkm::cont::LogCond(level, cond, __FILE__, __LINE__, __VA_ARGS__)

#define VTKM_LOG_S(level, ...) VTKM_LOG_IF_S(level, true, __VA_ARGS__)
#define VTKM_LOG_F(level, ...) VTKM_LOG_IF_F(level, true, __VA_ARGS__)

#define VTKM_LOG_SCOPE(level, ...) \
  vtkm::cont::detail::LogScope VTKM_ANONYMOUS_VARIABLE { level, __FILE__, __LINE__, __VA_ARGS__ }

#define VTKM_LOG_SCOPE_FUNCTION(level) VTKM_LOG_SCOPE(level, __func__)
#define VTKM_LOG_ALWAYS_S(level, ...) VTKM_LOG_S(level, __VA_ARGS__)


// Convenience macros:

// Cast success:
#define VTKM_LOG_CAST_SUCC(inObj, outObj)              \
  VTKM_LOG_F(vtkm::cont::LogLevel::Cast,               \
             "Cast succeeded: %s (%p) --> %s (%p)",    \
             vtkm::cont::TypeToString(inObj).c_str(),  \
             &inObj,                                   \
             vtkm::cont::TypeToString(outObj).c_str(), \
             &outObj)

// Cast failure:
#define VTKM_LOG_CAST_FAIL(inObj, outType)            \
  VTKM_LOG_F(vtkm::cont::LogLevel::Cast,              \
             "Cast failed: %s (%p) --> %s",           \
             vtkm::cont::TypeToString(inObj).c_str(), \
             &inObj,                                  \
             vtkm::cont::TypeToString<outType>().c_str())

// TryExecute failure
#define VTKM_LOG_TRYEXECUTE_FAIL(errorMessage, functorName, deviceId)                           \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "TryExecute encountered an error: " << errorMessage); \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "Failing functor: " << functorName);                  \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "Failing device: " << deviceId.GetName())

// Same, but disabling device:
#define VTKM_LOG_TRYEXECUTE_DISABLE(errorMessage, functorName, deviceId)                        \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "TryExecute encountered an error: " << errorMessage); \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "Failing functor: " << functorName);                  \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "Failing device: " << deviceId.GetName());            \
  VTKM_LOG_S(vtkm::cont::LogLevel::Error, "The failing device has been disabled.")

// Custom log level
#define VTKM_DEFINE_USER_LOG_LEVEL(name, offset)                                  \
  static constexpr vtkm::cont::LogLevel name = static_cast<vtkm::cont::LogLevel>( \
    static_cast<typename std::underlying_type<vtkm::cont::LogLevel>::type>(       \
      vtkm::cont::LogLevel::UserFirst) +                                          \
    offset %                                                                      \
      static_cast<typename std::underlying_type<vtkm::cont::LogLevel>::type>(     \
        vtkm::cont::LogLevel::UserLast))

#else // VTKM_ENABLE_LOGGING

#define VTKM_LOG_S(level, ...)
#define VTKM_LOG_F(level, ...)
#define VTKM_LOG_IF_S(level, cond, ...)
#define VTKM_LOG_IF_F(level, cond, ...)
#define VTKM_LOG_SCOPE(level, ...)
#define VTKM_LOG_SCOPE_FUNCTION(level)
#define VTKM_LOG_ERROR_CONTEXT(desc, data)
#define VTKM_LOG_CAST_SUCC(inObj, outObj)
#define VTKM_LOG_CAST_FAIL(inObj, outType)
#define VTKM_DEFINE_USER_LOG_LEVEL(name, offset)

// Always emitted. When logging is disabled, std::cerr is used.

#define VTKM_LOG_ALWAYS_S(level, ...)                   \
  (static_cast<int>(level) < 0 ? std::cerr : std::cout) \
    << vtkm::cont::GetLogLevelName(level) << ": " << __VA_ARGS__ << "\n"

// TryExecute failures are still important enough to log, but we just write to
// std::cerr when logging is disabled.
#define VTKM_LOG_TRYEXECUTE_FAIL(errorMessage, functorName, deviceId)             \
  std::cerr << "Error: TryExecute encountered an error: " << errorMessage << "\n" \
            << "\t- Failing functor: " << functorName << "\n"                     \
            << "\t- Failing device: " << deviceId.GetName() << "\n\n"
#define VTKM_LOG_TRYEXECUTE_DISABLE(errorMessage, functorName, deviceId)          \
  std::cerr << "Error: TryExecute encountered an error: " << errorMessage << "\n" \
            << "\t- Failing functor: " << functorName << "\n"                     \
            << "\t- Failing device: " << deviceId.GetName() << "\n"               \
            << "The failing device has been disabled.\n\n"

#endif // VTKM_ENABLE_LOGGING

namespace vtkm
{
namespace cont
{

/// Log levels for use with the logging macros.
enum class LogLevel
{
  /// A placeholder used to silence all logging. Do not actually log to
  /// this level.
  Off = -9, //loguru::Verbosity_OFF,

  /// Fatal errors that should abort execution.
  Fatal = -3, // loguru::Verbosity_FATAL,

  /// Important but non-fatal errors, such as device fail-over.
  Error = -2, // loguru::Verbosity_ERROR,

  /// Less important user errors, such as out-of-bounds parameters.
  Warn = -1, // loguru::Verbosity_WARNING,

  /// Information messages (detected hardware, etc) and temporary debugging
  /// output.
  Info = 0, //loguru::Verbosity_INFO,

  /// The first in a range of logging levels reserved for code that uses VTK-m.
  /// Internal VTK-m code will not log on these levels but will report these logs.
  UserFirst = 1,
  /// The last in a range of logging levels reserved for code that uses VTK-m.
  UserLast = 255,

  /// Information about which devices are enabled/disabled.
  DevicesEnabled,

  /// General timing data and algorithm flow information, such as filter
  /// execution, worklet dispatches, and device algorithm calls.
  Perf,

  /// Host-side resource allocations/frees (e.g. ArrayHandle control buffers).
  MemCont,

  /// Device-side resource allocations/frees (e.g ArrayHandle device buffers).
  MemExec,

  /// Transferring of data between a host and device.
  MemTransfer,

  /// Details on device-side kernel launches.
  KernelLaunches,

  /// Reports when a dynamic object is (or is not) resolved via a CastAndCall or other
  /// casting method.
  Cast,

  /// The first in a range of logging levels reserved for code that uses VTK-m.
  /// Internal VTK-m code will not log on these levels but will report these logs.
  /// These are used similarly to those in the UserFirst range but are at a lower
  /// precedence that also includes more verbose reporting from VTK-m.
  UserVerboseFirst = 1024,
  /// The last in a range of logging levels reserved for code that uses VTK-m.
  UserVerboseLast = 2047
};


/**
 * This shouldn't be called directly -- prefer calling vtkm::cont::Initialize,
 * which takes care of logging as well as other initializations.
 *
 * Initializes logging. Sets up custom log level and thread names. Parses any
 * "--vtkm-log-level [LogLevel]" arguments to set the stderr log level. This argument may
 * be either numeric, or the 4-character string printed in the output. Note that
 * loguru will consume the "--vtkm-log-level [LogLevel]" argument and shrink the arg list.
 *
 * If the parameterless overload is used, the `--vtkm-log-level` parsing is not used, but
 * other functionality should still work.
 *
 * @note This function is not threadsafe and should only be called from a single
 * thread (ideally the main thread).
 * @{
 */
VTKM_CONT_EXPORT
VTKM_CONT
void InitLogging(int& argc,
                 char* argv[],
                 const std::string& loggingFlag = "--vtkm-log-level",
                 const std::string& loggingEnv = "VTKM_LOG_LEVEL");
VTKM_CONT_EXPORT
VTKM_CONT
void InitLogging();
/**@}*/

/**
 * Set the range of log levels that will be printed to stderr. All levels
 * with an enum value less-than-or-equal-to \a level will be printed.
 * @{
 */
VTKM_CONT_EXPORT
VTKM_CONT
void SetStderrLogLevel(const char* verbosity);

VTKM_CONT_EXPORT
VTKM_CONT
void SetStderrLogLevel(vtkm::cont::LogLevel level);
/**@}*/

/**
 * Get the active highest log level that will be printed to stderr.
 */
VTKM_CONT_EXPORT
VTKM_CONT
vtkm::cont::LogLevel GetStderrLogLevel();

/**
 * Register a custom name to identify a log level. The name will be truncated
 * to 4 characters internally.
 *
 * Must not be called after InitLogging. Such calls will fail and log an error.
 *
 * There is no need to call this for the default vtkm::cont::LogLevels. They
 * are populated in InitLogging and will be overwritten.
 */
VTKM_CONT_EXPORT
VTKM_CONT
void SetLogLevelName(vtkm::cont::LogLevel level, const std::string& name);

/**
 * Get a human readable name for the log level. If a name has not been
 * registered via InitLogging or SetLogLevelName, the returned string just
 * contains the integer representation of the level.
 */
VTKM_CONT_EXPORT
VTKM_CONT
std::string GetLogLevelName(vtkm::cont::LogLevel level);

/**
 * Specifies a humman-readable name to identify the current thread in the log output.
 * @{
 */
VTKM_CONT_EXPORT
VTKM_CONT
void SetLogThreadName(const std::string& name);
VTKM_CONT_EXPORT
VTKM_CONT
std::string GetLogThreadName();
/**@}*/

// Per-thread error context, not currently used, undocumented....
VTKM_CONT_EXPORT
VTKM_CONT
std::string GetLogErrorContext();

/**
 * Returns a stacktrace on supported platforms.
 * Argument is the number of frames to skip (GetStackTrace and below are already
 * skipped).
 */
VTKM_CONT_EXPORT
VTKM_CONT
std::string GetStackTrace(vtkm::Int32 skip = 0);

//@{
/// Convert a size in bytes to a human readable string (such as "64 bytes",
/// "1.44 MiB", "128 GiB", etc). @a prec controls the fixed point precision
/// of the stringified number.
VTKM_CONT_EXPORT
VTKM_CONT
std::string GetHumanReadableSize(vtkm::UInt64 bytes, int prec = 2);

template <typename T>
VTKM_CONT inline std::string GetHumanReadableSize(T&& bytes, int prec = 2)
{
  return GetHumanReadableSize(static_cast<vtkm::UInt64>(std::forward<T>(bytes)), prec);
}
//@}

//@{
/// Returns "%1 (%2 bytes)" where %1 is the result from GetHumanReadableSize
/// and %2 is the exact number of bytes.
VTKM_CONT_EXPORT
VTKM_CONT
std::string GetSizeString(vtkm::UInt64 bytes, int prec = 2);

template <typename T>
VTKM_CONT inline std::string GetSizeString(T&& bytes, int prec = 2)
{
  return GetSizeString(static_cast<vtkm::UInt64>(std::forward<T>(bytes)), prec);
}
//@}

/**
 * Use RTTI information to retrieve the name of the type T. If logging is
 * enabled and the platform supports it, the type name will also be demangled.
 * @{
 */
VTKM_CONT_EXPORT VTKM_CONT std::string TypeToString(const std::type_info& t);
VTKM_CONT_EXPORT VTKM_CONT std::string TypeToString(const std::type_index& t);
template <typename T>
inline VTKM_CONT std::string TypeToString()
{
  return TypeToString(typeid(T));
}
template <typename T>
inline VTKM_CONT std::string TypeToString(const T&)
{
  return TypeToString(typeid(T));
}
/**@}*/

#ifdef VTKM_ENABLE_LOGGING

/**
 * \brief Conditionally logs a message with a printf-like format.
 *
 * \param level  Desired LogLevel value for the log message.
 * \param cond   When false this function is no-op.
 * \param file   The source file where the log entry was genearted.
 * \param line   The line in the source file where the log entry was generated.
 * \param format Printf like format string.
 */
VTKM_CONT_EXPORT
VTKM_CONT
void LogCond(LogLevel level, bool cond, const char* file, unsigned line, const char* format...);

namespace detail
{

/**
 * \brief Logs a scoped message with a printf-like format.
 *
 * The indentation level will be determined based on its LogLevel and it will
 * print out its wall time upon exiting its scope. The scope starts from when
 * the object is created to when it is destroyed.
 */
class VTKM_CONT_EXPORT LogScope
{
  struct InternalStruct;
  std::unique_ptr<InternalStruct> Internals;

public:
  /*
   * \param level  Desired LogLevel value for the log message.
   * \param cond   When false this function is no-op.
   * \param format Printf like format string.
   */
  VTKM_CONT
  LogScope(LogLevel level, const char* file, unsigned line, const char* format...);

  VTKM_CONT ~LogScope();
};


} // namespace detail

/**
 * \brief Conditionally logs a message with a stream-like interface.
 *
 * Messages are flushed to output by the destructor.
 */
struct VTKM_CONT_EXPORT LogCondStream
{
  VTKM_CONT
  LogCondStream(LogLevel level, bool cond, const char* file, int line)
    : Level(level)
    , Condition(cond)
    , File(file)
    , Line(line)
  {
  }

  VTKM_CONT
  ~LogCondStream() noexcept(false);

  template <typename T>
  VTKM_CONT LogCondStream& operator<<(const T& in)
  {
    SStream << in;
    return *this;
  }

  VTKM_CONT
  LogCondStream& operator<<(std::ostream& (*f)(std::ostream&))
  {
    f(SStream);
    return *this;
  }

private:
  LogLevel Level;
  bool Condition;
  const char* File;
  int Line;
  std::ostringstream SStream;
};
#endif // VTKM_ENABLE_LOGGING

}
} // end namespace vtkm::cont

#endif // vtk_m_cont_Logging_h
