#pragma once
#include "utils/check.hpp"
#include "utils/types.hpp"

#define casting(T, v) llaisys::utils::cast<T>(v)
#define recast(T, v) reinterpret_cast<T>(v)