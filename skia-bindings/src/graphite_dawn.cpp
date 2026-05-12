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
#include "dawn/dawn_proc.h"
#include "dawn/native/DawnNative.h"
#include "webgpu/webgpu.h"
#include "webgpu/webgpu_cpp.h"

#include <mutex>

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

namespace {

void adapter_callback(WGPURequestAdapterStatus status,
                      WGPUAdapter adapter,
                      WGPUStringView /*message*/,
                      void* userdata1,
                      void* /*userdata2*/) {
    if (status == WGPURequestAdapterStatus_Success) {
        *static_cast<WGPUAdapter*>(userdata1) = adapter;
    }
}

void device_callback(WGPURequestDeviceStatus status,
                     WGPUDevice device,
                     WGPUStringView /*message*/,
                     void* userdata1,
                     void* /*userdata2*/) {
    if (status == WGPURequestDeviceStatus_Success) {
        *static_cast<WGPUDevice*>(userdata1) = device;
    }
}

} // namespace

// Creates a default Dawn instance, requests an adapter and device, and
// returns the resulting raw handles via the out-params. Dawn picks the
// platform's preferred backend (Vulkan on Linux/Android, Metal on macOS/iOS,
// D3D12 on Windows). On success the caller owns one reference of each handle
// and must release them.
extern "C" bool C_SkgpuGraphite_DawnDefaultSetup(
        WGPUInstance* outInstance,
        WGPUDevice* outDevice,
        WGPUQueue* outQueue) {
    *outInstance = nullptr;
    *outDevice = nullptr;
    *outQueue = nullptr;

    // The `dawn_proc` dispatcher routes all wgpu* calls through a function-
    // pointer table that is null until we point it at Dawn's native procs.
    // We do this exactly once per process; it is safe to set the same table
    // from multiple threads but `dawnProcSetProcs` itself is not thread-safe,
    // so guard with std::once_flag.
    static std::once_flag procsOnce;
    std::call_once(procsOnce, []() {
        static DawnProcTable procs = dawn::native::GetProcs();
        dawnProcSetProcs(&procs);
    });

    // Request the TimedWaitAny instance feature so we can block on adapter /
    // device futures with a real timeout (UINT64_MAX = wait forever).
    WGPUInstanceFeatureName requiredFeatures[] = { WGPUInstanceFeatureName_TimedWaitAny };
    WGPUInstanceDescriptor desc = {};
    desc.requiredFeatureCount = 1;
    desc.requiredFeatures = requiredFeatures;

    WGPUInstance instance = wgpuCreateInstance(&desc);
    if (!instance) {
        return false;
    }

    WGPUAdapter adapter = nullptr;
    WGPURequestAdapterCallbackInfo adapterCbInfo = {};
    adapterCbInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    adapterCbInfo.callback = adapter_callback;
    adapterCbInfo.userdata1 = &adapter;
    WGPUFuture adapterFuture = wgpuInstanceRequestAdapter(instance, nullptr, adapterCbInfo);
    WGPUFutureWaitInfo adapterWait{adapterFuture, 0};
    wgpuInstanceWaitAny(instance, 1, &adapterWait, UINT64_MAX);
    if (!adapter) {
        wgpuInstanceRelease(instance);
        return false;
    }

    WGPUDevice device = nullptr;
    WGPURequestDeviceCallbackInfo deviceCbInfo = {};
    deviceCbInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    deviceCbInfo.callback = device_callback;
    deviceCbInfo.userdata1 = &device;
    WGPUFuture deviceFuture = wgpuAdapterRequestDevice(adapter, nullptr, deviceCbInfo);
    WGPUFutureWaitInfo deviceWait{deviceFuture, 0};
    wgpuInstanceWaitAny(instance, 1, &deviceWait, UINT64_MAX);
    wgpuAdapterRelease(adapter);
    if (!device) {
        wgpuInstanceRelease(instance);
        return false;
    }

    *outInstance = instance;
    *outDevice = device;
    *outQueue = wgpuDeviceGetQueue(device);
    return true;
}
