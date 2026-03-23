#include "iluvatar_resource.cuh"

namespace llaisys::device::iluvatar {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_ILUVATAR, device_id) {}

} // namespace llaisys::device::iluvatar
