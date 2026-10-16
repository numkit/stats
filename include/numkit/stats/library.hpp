/// @file library.hpp
/// @ingroup group_stats
#pragma once

namespace numkit {

/// @addtogroup group_stats
/// @{
class PluginContext;

class StatsLibrary
{
public:
    static void install(PluginContext &engine);
};

/// @}
} // namespace numkit
