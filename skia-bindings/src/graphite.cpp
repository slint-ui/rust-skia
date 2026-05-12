// Wrappers for `skgpu::graphite::*` core types. Backend-agnostic — factories
// for specific GPU backends (Dawn, Metal, Vulkan) live in their own files.

#ifndef SK_GRAPHITE
    #define SK_GRAPHITE
#endif

#include "bindings.h"

#include "include/core/SkImageInfo.h"
#include "include/core/SkSurface.h"
#include "include/core/SkSurfaceProps.h"
#include "include/gpu/GpuTypes.h"
#include "include/gpu/graphite/Context.h"
#include "include/gpu/graphite/ContextOptions.h"
#include "include/gpu/graphite/GraphiteTypes.h"
#include "include/gpu/graphite/Recorder.h"
#include "include/gpu/graphite/Recording.h"
#include "include/gpu/graphite/Surface.h"

namespace skgr = skgpu::graphite;

// Force bindgen to surface a few small POD types that would otherwise be elided
// because nothing else exposes them via an extern "C" signature.
extern "C" void C_SkgpuGraphiteUnreferencedTypes(
        skgr::InsertStatus::V*,
        skgr::SyncToCpu*,
        skgpu::CallbackResult*,
        skgpu::BackendApi*,
        skgr::SubmitInfo*) {}

//
// skgpu::graphite::Context
//

extern "C" void C_SkgpuGraphiteContext_delete(skgr::Context* ctx) {
    delete ctx;
}

extern "C" skgpu::BackendApi C_SkgpuGraphiteContext_backend(const skgr::Context* ctx) {
    return ctx->backend();
}

extern "C" skgr::Recorder* C_SkgpuGraphiteContext_makeRecorder(
        skgr::Context* ctx,
        const skgr::RecorderOptions* options) {
    return ctx->makeRecorder(*options).release();
}

extern "C" bool C_SkgpuGraphiteContext_submit(
        skgr::Context* ctx,
        const skgr::SubmitInfo* submitInfo) {
    return ctx->submit(*submitInfo);
}

// Inserts a Recording for replay against an optional target surface. The
// Recording is not consumed by this call — the caller still owns it and must
// drop it after submission (Skia internally retains what it needs).
extern "C" skgr::InsertStatus::V C_SkgpuGraphite_Context_insertRecording(
        skgr::Context* ctx,
        skgr::Recording* recording,
        SkSurface* targetSurface) {
    skgr::InsertRecordingInfo info;
    info.fRecording = recording;
    info.fTargetSurface = targetSurface;
    skgr::InsertStatus status = ctx->insertRecording(info);
    return static_cast<skgr::InsertStatus::V>(status);
}

extern "C" bool C_SkgpuGraphiteContext_hasUnfinishedGpuWork(const skgr::Context* ctx) {
    return ctx->hasUnfinishedGpuWork();
}

extern "C" bool C_SkgpuGraphiteContext_hasPendingGPUWork(const skgr::Context* ctx) {
    return ctx->hasPendingGPUWork();
}

extern "C" void C_SkgpuGraphiteContext_checkAsyncWorkCompletion(skgr::Context* ctx) {
    ctx->checkAsyncWorkCompletion();
}

extern "C" bool C_SkgpuGraphiteContext_isDeviceLost(const skgr::Context* ctx) {
    return ctx->isDeviceLost();
}

extern "C" int C_SkgpuGraphiteContext_maxTextureSize(const skgr::Context* ctx) {
    return ctx->maxTextureSize();
}

extern "C" bool C_SkgpuGraphiteContext_supportsProtectedContent(const skgr::Context* ctx) {
    return ctx->supportsProtectedContent();
}

extern "C" void C_SkgpuGraphiteContext_freeGpuResources(skgr::Context* ctx) {
    ctx->freeGpuResources();
}

extern "C" size_t C_SkgpuGraphiteContext_currentBudgetedBytes(const skgr::Context* ctx) {
    return ctx->currentBudgetedBytes();
}

extern "C" size_t C_SkgpuGraphiteContext_currentPurgeableBytes(const skgr::Context* ctx) {
    return ctx->currentPurgeableBytes();
}

extern "C" size_t C_SkgpuGraphiteContext_maxBudgetedBytes(const skgr::Context* ctx) {
    return ctx->maxBudgetedBytes();
}

extern "C" void C_SkgpuGraphiteContext_setMaxBudgetedBytes(skgr::Context* ctx, size_t bytes) {
    ctx->setMaxBudgetedBytes(bytes);
}

//
// skgpu::graphite::Recorder
//

extern "C" void C_SkgpuGraphiteRecorder_delete(skgr::Recorder* r) {
    delete r;
}

extern "C" skgpu::BackendApi C_SkgpuGraphiteRecorder_backend(const skgr::Recorder* r) {
    return r->backend();
}

extern "C" skgr::Recording* C_SkgpuGraphiteRecorder_snap(skgr::Recorder* r) {
    return r->snap().release();
}

extern "C" int C_SkgpuGraphiteRecorder_maxTextureSize(const skgr::Recorder* r) {
    return r->maxTextureSize();
}

extern "C" void C_SkgpuGraphiteRecorder_freeGpuResources(skgr::Recorder* r) {
    r->freeGpuResources();
}

extern "C" size_t C_SkgpuGraphiteRecorder_currentBudgetedBytes(const skgr::Recorder* r) {
    return r->currentBudgetedBytes();
}

extern "C" size_t C_SkgpuGraphiteRecorder_currentPurgeableBytes(const skgr::Recorder* r) {
    return r->currentPurgeableBytes();
}

extern "C" size_t C_SkgpuGraphiteRecorder_maxBudgetedBytes(const skgr::Recorder* r) {
    return r->maxBudgetedBytes();
}

extern "C" void C_SkgpuGraphiteRecorder_setMaxBudgetedBytes(skgr::Recorder* r, size_t bytes) {
    r->setMaxBudgetedBytes(bytes);
}

//
// skgpu::graphite::Recording
//

extern "C" void C_SkgpuGraphiteRecording_delete(skgr::Recording* rec) {
    delete rec;
}

//
// Options structs
//

extern "C" void C_SkgpuGraphiteContextOptions_Construct(skgr::ContextOptions* uninitialized) {
    new(uninitialized) skgr::ContextOptions();
}

extern "C" void C_SkgpuGraphiteContextOptions_destruct(skgr::ContextOptions* opts) {
    opts->~ContextOptions();
}

extern "C" void C_SkgpuGraphiteRecorderOptions_Construct(skgr::RecorderOptions* uninitialized) {
    new(uninitialized) skgr::RecorderOptions();
}

extern "C" void C_SkgpuGraphiteRecorderOptions_destruct(skgr::RecorderOptions* opts) {
    opts->~RecorderOptions();
}

//
// SkSurfaces:: factories for Graphite
//

// Allocates a renderable surface backed by a Graphite texture. Caller takes
// ownership of the returned SkSurface (refcounted) and is responsible for
// dropping it.
extern "C" SkSurface* C_SkgpuGraphite_Surfaces_RenderTarget(
        skgr::Recorder* recorder,
        const SkImageInfo* imageInfo,
        skgpu::Mipmapped mipmapped,
        const SkSurfaceProps* surfaceProps) {
    sk_sp<SkSurface> surface = SkSurfaces::RenderTarget(
            recorder, *imageInfo, mipmapped, surfaceProps);
    return surface.release();
}
