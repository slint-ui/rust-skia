// Dawn factory for Graphite Contexts.

#ifndef SK_GRAPHITE
    #define SK_GRAPHITE
#endif
#ifndef SK_DAWN
    #define SK_DAWN
#endif

#include "bindings.h"

#include "include/gpu/graphite/Context.h"
#include "include/gpu/graphite/ContextOptions.h"
#include "include/gpu/graphite/dawn/DawnBackendContext.h"
#include "webgpu/webgpu.h"
#include "webgpu/webgpu_cpp.h"

// Creates a Graphite Context backed by Dawn. The caller passes raw WebGPU C
// handles; this function bumps each handle's reference count so the caller
// retains ownership of its own references after the call. The new Context
// owns its own refs and releases them when destroyed.
extern "C" skgpu::graphite::Context* C_SkgpuGraphite_ContextFactory_MakeDawn(
        WGPUInstance instance,
        WGPUDevice device,
        WGPUQueue queue,
        const skgpu::graphite::ContextOptions* options) {
    wgpuInstanceAddRef(instance);
    wgpuDeviceAddRef(device);
    wgpuQueueAddRef(queue);

    skgpu::graphite::DawnBackendContext backendCtx;
    backendCtx.fInstance = wgpu::Instance::Acquire(instance);
    backendCtx.fDevice = wgpu::Device::Acquire(device);
    backendCtx.fQueue = wgpu::Queue::Acquire(queue);

    return skgpu::graphite::ContextFactory::MakeDawn(backendCtx, *options).release();
}
