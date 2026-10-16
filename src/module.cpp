// src/module.cpp
//
// Plugin entry point for numkit_stats dynamic module.
// Exports numkit_plugin_init conforming to NUMKIT_PLUGIN_ABI_VERSION.

#include <numkit/sys/plugin.hpp>
#include <numkit/stats/library.hpp>

extern "C" NUMKIT_PLUGIN_EXPORT bool numkit_plugin_init(numkit::PluginContext &ctx)
{
    if (ctx.abiVersion() != NUMKIT_PLUGIN_ABI_VERSION) {
        return false;
    }
    numkit::StatsLibrary::install(ctx);
    return true;
}
