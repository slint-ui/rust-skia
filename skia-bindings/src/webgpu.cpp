// Surfaces the WebGPU C API for bindgen.
//
// When the `graphite` feature is enabled, Skia is built with Dawn statically
// linked. Dawn ships the standard `webgpu.h` C header at
// `third_party/externals/dawn/include`, which the build system adds to the
// include path. Bindgen parses this header to expose the `WGPU*` handle types,
// enums, descriptor structs, and `wgpu*` functions to Rust.
//
// Nothing else is needed here — the WebGPU C ABI is already `extern "C"`.

#include "webgpu/webgpu.h"

extern "C" void C_WebGPU_Unused(void) {}
