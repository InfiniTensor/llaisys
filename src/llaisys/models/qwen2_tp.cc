// Include the actual TP implementation to ensure symbols are exported
// This file is compiled as part of the shared library

// Define LLAISYS_BUILDING_SHARED to ensure proper export
#define LLAISYS_BUILDING_SHARED

// Include the implementation file
#include "../../models/qwen2/qwen2_tp.cpp"
