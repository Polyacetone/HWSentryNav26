#pragma once

// No discard macro to be compatible with pre c++17.

#ifndef UBS_NO_DISCARD
#if __cplusplus >= 201703L
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define UBS_NO_DISCARD [[nodiscard]]
#else
#define UBS_NO_DISCARD
#endif
#endif
