//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include <vtkm/cont/Initialize.h>

#include <vtkm/cont/Logging.h>
#include <vtkm/cont/RuntimeDeviceTracker.h>
#include <vtkm/cont/internal/OptionParser.h>
#include <vtkm/cont/internal/OptionParserArguments.h>

#include <vtkm/thirdparty/diy/environment.h>

#include <cstdlib>
#include <memory>
#include <sstream>

namespace opt = vtkm::cont::internal::option;

namespace
{

struct VtkmDeviceArg : public opt::Arg
{
  static opt::ArgStatus IsDevice(const opt::Option& option, bool msg)
  {
    // Device must be specified if option is present:
    if (option.arg == nullptr)
    {
      if (msg)
      {
        VTKM_LOG_ALWAYS_S(vtkm::cont::LogLevel::Error,
                          "Missing device after option '"
                            << std::string(option.name, static_cast<size_t>(option.namelen))
                            << "'.\nValid devices are: " << VtkmDeviceArg::GetValidDeviceNames()
                            << "\n");
      }
      return opt::ARG_ILLEGAL;
    }

    auto id = vtkm::cont::make_DeviceAdapterId(option.arg);

    if (!VtkmDeviceArg::DeviceIsAvailable(id))
    {
      VTKM_LOG_ALWAYS_S(vtkm::cont::LogLevel::Error,
                        "Unavailable device specificed after option '"
                          << std::string(option.name, static_cast<size_t>(option.namelen)) << "': '"
                          << option.arg << "'.\nValid devices are: "
                          << VtkmDeviceArg::GetValidDeviceNames() << "\n");
      return opt::ARG_ILLEGAL;
    }

    return opt::ARG_OK;
  }

  static bool DeviceIsAvailable(vtkm::cont::DeviceAdapterId id)
  {
    if (id == vtkm::cont::DeviceAdapterTagAny{})
    {
      return true;
    }

    if (id.GetValue() <= 0 || id.GetValue() >= VTKM_MAX_DEVICE_ADAPTER_ID ||
        id == vtkm::cont::DeviceAdapterTagUndefined{})
    {
      return false;
    }

    auto& tracker = vtkm::cont::GetRuntimeDeviceTracker();
    bool result = false;
    try
    {
      result = tracker.CanRunOn(id);
    }
    catch (...)
    {
      result = false;
    }
    return result;
  }
  static std::string GetValidDeviceNames()
  {
    std::ostringstream names;
    names << "\"Any\" ";

    for (vtkm::Int8 i = 0; i < VTKM_MAX_DEVICE_ADAPTER_ID; ++i)
    {
      auto id = vtkm::cont::make_DeviceAdapterId(i);
      if (VtkmDeviceArg::DeviceIsAvailable(id))
      {
        names << "\"" << id.GetName() << "\" ";
      }
    }
    return names.str();
  }
};

} // namespace

namespace vtkm
{
namespace cont
{

VTKM_CONT
InitializeResult Initialize(int& argc, char* argv[], InitializeOptions opts)
{
  InitializeResult config;
  const std::string loggingFlagName = "vtkm-log-level";
  const std::string loggingFlag = "--" + loggingFlagName;
  const std::string loggingHelp = "  " + loggingFlag +
    " <#|INFO|WARNING|ERROR|FATAL|OFF> \tSpecify a log level (when logging is enabled).";

  // initialize logging and diy first -- they'll pop off the options they consume:
  if (argc == 0 || argv == nullptr)
  {
    vtkm::cont::InitLogging();
  }
  else
  {
    vtkm::cont::InitLogging(argc, argv, loggingFlag, "VTKM_LOG_LEVEL");
  }
  if (!vtkmdiy::mpi::environment::initialized())
  {
    if (argc == 0 || argv == nullptr)
    {
      // If initialized, will be deleted on program exit (calling MPI_Finalize if necessary)
      static vtkmdiy::mpi::environment diyEnvironment;
    }
    else
    {
      // If initialized, will be deleted on program exit (calling MPI_Finalize if necessary)
      static vtkmdiy::mpi::environment diyEnvironment(argc, argv);
    }
  }

  { // Parse VTKm options
    std::vector<opt::Descriptor> usage;
    if ((opts & InitializeOptions::AddHelp) != InitializeOptions::None)
    {
      // Because we have the AddHelp option, we will add both --help and --vtkm-help to
      // the list of arguments. Use the first entry for introduction on the usage.
      usage.push_back(
        { opt::OptionIndex::HELP, 0, "", "vtkm-help", opt::Arg::None, "Usage information:\n" });
      usage.push_back({ opt::OptionIndex::HELP,
                        0,
                        "h",
                        "help",
                        opt::Arg::None,
                        "  --help, --vtkm-help, -h \tPrint usage information." });
    }
    else
    {
      usage.push_back({ opt::OptionIndex::HELP,
                        0,
                        "",
                        "vtkm-help",
                        opt::Arg::None,
                        "  --vtkm-help \tPrint usage information." });
    }
    usage.push_back(
      { opt::OptionIndex::DEVICE,
        0,
        "",
        "vtkm-device",
        VtkmDeviceArg::IsDevice,
        "  --vtkm-device <dev> \tForce device to dev. Omit device to list available devices." });
    usage.push_back({ opt::OptionIndex::LOGLEVEL,
                      0,
                      "",
                      loggingFlagName.c_str(),
                      opt::VtkmArg::Required,
                      loggingHelp.c_str() });

    // Bring in extra args used by the runtime device configuration options
    vtkm::cont::internal::RuntimeDeviceConfigurationOptions runtimeDeviceOptions(usage);

    // Required to collect unknown arguments.
    usage.push_back({ opt::OptionIndex::UNKNOWN, 0, "", "", opt::VtkmArg::UnknownOption, "" });
    usage.push_back({ 0, 0, 0, 0, 0, 0 });

    {
      std::stringstream streamBuffer;
      opt::printUsage(streamBuffer, usage.data());
      config.Usage = streamBuffer.str();
      // Remove trailing newline as one more than we want is added.
      config.Usage = config.Usage.substr(0, config.Usage.length() - 1);
    }

    // Remove argv[0] (executable name) if present:
    int vtkmArgc = argc > 0 ? argc - 1 : 0;
    char** vtkmArgv = vtkmArgc > 0 ? argv + 1 : argv;

    opt::Stats stats(usage.data(), vtkmArgc, vtkmArgv);
    std::unique_ptr<opt::Option[]> options{ new opt::Option[stats.options_max] };
    std::unique_ptr<opt::Option[]> buffer{ new opt::Option[stats.buffer_max] };
    opt::Parser parse(usage.data(), vtkmArgc, vtkmArgv, options.get(), buffer.get());

    if (parse.error())
    {
      std::cerr << config.Usage;
      exit(1);
    }

    if (options[opt::OptionIndex::HELP])
    {
      std::cerr << config.Usage;
      exit(0);
    }

    // The RuntimeDeviceConfiguration must be completed before calling GetRuntimeDeviceTracker()
    // for all the devices. This is because GetRuntimeDeviceTracker will construct a given
    // device's DeviceAdapterRuntimeDetector to determine if it exists and this constructor may
    // call `GetRuntimeConfiguration` for the specific device in order to query things such as
    // available threads/devices.
    {
      runtimeDeviceOptions.Initialize(options.get());
      vtkm::cont::RuntimeDeviceInformation runtimeDevice;
      runtimeDevice.GetRuntimeConfiguration(
        vtkm::cont::DeviceAdapterTagAny{}, runtimeDeviceOptions, argc, argv);
    }

    // Check for device on command line.
    if (options[opt::OptionIndex::DEVICE])
    {
      const char* arg = options[opt::OptionIndex::DEVICE].arg;
      config.Device = vtkm::cont::make_DeviceAdapterId(arg);
    }
    // If not on command line, check for device in environment variable.
    if (config.Device == vtkm::cont::DeviceAdapterTagUndefined{})
    {
      const char* deviceEnv = std::getenv("VTKM_DEVICE");
      if (deviceEnv != nullptr)
      {
        auto id = vtkm::cont::make_DeviceAdapterId(std::getenv("VTKM_DEVICE"));
        if (VtkmDeviceArg::DeviceIsAvailable(id))
        {
          config.Device = id;
        }
        else
        {
          // Got invalid device. Log an error, but continue to do the default action for
          // the device (i.e., ignore the environment variable setting).
          VTKM_LOG_S(vtkm::cont::LogLevel::Error,
                     "Invalid device `"
                       << deviceEnv
                       << "` specified in VTKM_DEVICE environment variable. Ignoring.");
          VTKM_LOG_S(vtkm::cont::LogLevel::Error,
                     "Valid devices are: " << VtkmDeviceArg::GetValidDeviceNames());
        }
      }
    }
    // If still not defined, check to see if "any" device should be added.
    if ((config.Device == vtkm::cont::DeviceAdapterTagUndefined{}) &&
        (opts & InitializeOptions::DefaultAnyDevice) != InitializeOptions::None)
    {
      config.Device = vtkm::cont::DeviceAdapterTagAny{};
    }
    // Set the state for the device selected.
    if (config.Device == vtkm::cont::DeviceAdapterTagUndefined{})
    {
      if ((opts & InitializeOptions::RequireDevice) != InitializeOptions::None)
      {
        auto devices = VtkmDeviceArg::GetValidDeviceNames();
        VTKM_LOG_S(vtkm::cont::LogLevel::Fatal, "Device not given on command line.");
        std::cerr << "Target device must be specified via --vtkm-device.\n"
                     "Valid devices: "
                  << devices << std::endl;
        if ((opts & InitializeOptions::AddHelp) != InitializeOptions::None)
        {
          std::cerr << config.Usage;
        }
        exit(1);
      }
      else
      {
        // No device specified. Do nothing and let VTK-m decide what it is going to do.
      }
    }
    else if (config.Device == vtkm::cont::DeviceAdapterTagAny{})
    {
      vtkm::cont::GetRuntimeDeviceTracker().Reset();
    }
    else
    {
      vtkm::cont::GetRuntimeDeviceTracker().ForceDevice(config.Device);
    }


    for (const opt::Option* opt = options[opt::OptionIndex::UNKNOWN]; opt != nullptr;
         opt = opt->next())
    {
      if ((opts & InitializeOptions::ErrorOnBadOption) != InitializeOptions::None)
      {
        std::cerr << "Unknown option: " << opt->name << std::endl;
        if ((opts & InitializeOptions::AddHelp) != InitializeOptions::None)
        {
          std::cerr << config.Usage;
        }
        exit(1);
      }
    }

    for (int nonOpt = 0; nonOpt < parse.nonOptionsCount(); ++nonOpt)
    {
      VTKM_LOG_S(vtkm::cont::LogLevel::Info,
                 "Unknown argument to Initialize: " << parse.nonOption(nonOpt) << "\n");
      if ((opts & InitializeOptions::ErrorOnBadArgument) != InitializeOptions::None)
      {
        std::cerr << "Unknown argument: " << parse.nonOption(nonOpt) << std::endl;
        if ((opts & InitializeOptions::AddHelp) != InitializeOptions::None)
        {
          std::cerr << config.Usage;
        }
        exit(1);
      }
    }

    // Now go back through the arg list and remove anything that is not in the list of
    // unknown options or non-option arguments.
    int destArg = 1;
    for (int srcArg = 1; srcArg < argc; ++srcArg)
    {
      std::string thisArg{ argv[srcArg] };
      bool copyArg = false;

      // Special case: "--" gets removed by optionparser but should be passed.
      if (thisArg == "--")
      {
        copyArg = true;
      }
      for (const opt::Option* opt = options[opt::OptionIndex::UNKNOWN]; !copyArg && opt != nullptr;
           opt = opt->next())
      {
        if (thisArg == opt->name)
        {
          copyArg = true;
        }
        if ((opt->arg != nullptr) && (thisArg == opt->arg))
        {
          copyArg = true;
        }
        // Special case: optionparser sometimes removes a single "-" from an option
        if (thisArg.substr(1) == opt->name)
        {
          copyArg = true;
        }
      }
      for (int nonOpt = 0; !copyArg && nonOpt < parse.nonOptionsCount(); ++nonOpt)
      {
        if (thisArg == parse.nonOption(nonOpt))
        {
          copyArg = true;
        }
      }
      if (copyArg)
      {
        if (destArg != srcArg)
        {
          argv[destArg] = argv[srcArg];
        }
        ++destArg;
      }
    }
    argc = destArg;
  }

  return config;
}

VTKM_CONT
InitializeResult Initialize()
{
  int argc = 0;
  char** argv = nullptr;
  return Initialize(argc, argv);
}
}
} // end namespace vtkm::cont
