#pragma once
// Use friend test macro only if gtest is available.

#if __has_include(<gtest/gtest_prod.h>)
#include <gtest/gtest_prod.h>
#else
#ifndef FRIEND_TEST
#define FRIEND_TEST(test_case_name, test_name)
#endif
#endif
