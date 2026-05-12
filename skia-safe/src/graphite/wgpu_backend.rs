//! Experimental wgpu-backed proc table for Graphite.
//!
//! When the `wgpu` Cargo feature is enabled, this module exposes
//! [`wgpu_proc_table`], a [`DawnProcTable`](super::dawn::DawnProcTable) whose
//! entries route every `wgpu*` C call into the Rust `wgpu` crate. Combined
//! with [`super::dawn::install_proc_table`], this lets Skia/Graphite render
//! through the same WebGPU implementation as the rest of an application that
//! already uses `wgpu`.
//!
//! # Status
//!
//! **Incomplete.** The WebGPU C ABI surface is large (~270 functions). Only
//! a handful are implemented; everything else aborts the process with the
//! function name on first call. This module exists primarily as the
//! scaffolding for filling out coverage incrementally — landing it lets
//! Skia tell us, by aborting, which functions it actually needs.
//!
//! # Coexistence with Dawn
//!
//! Enabling `wgpu` does *not* unlink Dawn — `libdawn_combined.a` remains in
//! the static library. The proc-table mechanism is the only way Skia's
//! `wgpu*` calls are routed elsewhere. Users who want Dawn-native rendering
//! can simply not call `install_proc_table(&wgpu_proc_table())` and the
//! default Dawn path continues to work.

use crate::graphite::dawn::DawnProcTable;

/// Aborts the process with a message identifying the WGPU function that
/// hasn't been implemented yet. Used as the default proc-table entry until
/// real thunks land.
///
/// We deliberately call `std::process::abort()` rather than panicking: a
/// panic that crosses an FFI boundary back into Dawn's C dispatcher is UB.
fn unimplemented_stub(name: &'static str) -> ! {
    eprintln!("skia_safe::graphite::wgpu_backend: unimplemented WebGPU call `{name}`");
    std::process::abort();
}

/// Generates an `unsafe extern "C" fn` stub for a given WGPU function name
/// and signature. The stub aborts when called.
macro_rules! stub {
    ($name:ident ( $($arg:ident : $arg_ty:ty),* $(,)? ) -> $ret:ty) => {{
        unsafe extern "C" fn stub( $( $arg : $arg_ty ),* ) -> $ret {
            $( let _ = $arg; )*
            $crate::graphite::wgpu_backend::unimplemented_stub(stringify!($name))
        }
        Some(stub as _)
    }};
    ($name:ident ( $($arg:ident : $arg_ty:ty),* $(,)? )) => {{
        unsafe extern "C" fn stub( $( $arg : $arg_ty ),* ) {
            $( let _ = $arg; )*
            $crate::graphite::wgpu_backend::unimplemented_stub(stringify!($name))
        }
        Some(stub as _)
    }};
}

/// Builds a [`DawnProcTable`] whose entries route to wgpu.
///
/// Currently every entry is an abort-on-call stub. Fill out by replacing
/// individual fields with real thunks against the `wgpu` crate API.
pub fn wgpu_proc_table() -> DawnProcTable {
    // SAFETY: A zeroed DawnProcTable has all function pointers set to `None`
    // (since `Option<unsafe extern "C" fn ...>` is layout-compatible with a
    // null pointer in the `None` case). The fields we explicitly populate
    // below replace those nulls.
    let mut table: DawnProcTable = unsafe { core::mem::zeroed() };

    populate_stubs(&mut table);

    table
}

/// Populates the known-needed-soon entries with abort-stubs. As real thunks
/// land they replace entries here. Functions Skia never calls can stay
/// `None`; if Skia ever does call one, the dispatcher null-dereferences and
/// crashes — slightly less friendly than the stub's abort message, but the
/// crash backtrace still points at the missing function.
fn populate_stubs(table: &mut DawnProcTable) {
    use skia_bindings as sb;

    // Instance lifecycle. These are the first calls Skia makes during
    // DawnDefaultSetup, so they're a natural starting point for real thunks.
    table.createInstance = stub!(
        wgpuCreateInstance(descriptor: *const sb::WGPUInstanceDescriptor) -> sb::WGPUInstance
    );
    table.instanceAddRef = stub!(wgpuInstanceAddRef(instance: sb::WGPUInstance));
    table.instanceRelease = stub!(wgpuInstanceRelease(instance: sb::WGPUInstance));
    table.instanceRequestAdapter = stub!(
        wgpuInstanceRequestAdapter(
            instance: sb::WGPUInstance,
            options: *const sb::WGPURequestAdapterOptions,
            callbackInfo: sb::WGPURequestAdapterCallbackInfo,
        ) -> sb::WGPUFuture
    );
    table.instanceWaitAny = stub!(
        wgpuInstanceWaitAny(
            instance: sb::WGPUInstance,
            futureCount: usize,
            futures: *mut sb::WGPUFutureWaitInfo,
            timeoutNS: u64,
        ) -> sb::WGPUWaitStatus
    );
    table.instanceProcessEvents = stub!(wgpuInstanceProcessEvents(instance: sb::WGPUInstance));
}
