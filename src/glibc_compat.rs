// glibc < 2.38 link-compatibility shims for the bundled ONNX Runtime (#526).
//
// The prebuilt `libonnxruntime.a` that the `ort` crate downloads (pyke CDN,
// ms@1.24.2) is compiled on Ubuntu 24.04, whose glibc (>= 2.38) redirects the
// C23 `strto*` family to new `__isoc23_*` symbols. Hosts on older glibc —
// notably Ubuntu 22.04 / glibc 2.35, the dominant cloud + CI base image —
// don't export those symbols, so the DEFAULT `cargo build` (bundled-embeddings
// on by default) died at link time with:
//
//     rust-lld: error: undefined symbol: __isoc23_strtoll
//
// The archive references exactly six such symbols (verified by scanning the
// artifact): __isoc23_strtol, __isoc23_strtoll, __isoc23_strtoul,
// __isoc23_strtoull, __isoc23_strtoll_l, __isoc23_strtoull_l. Each is
// semantically the C23 edition of the classic function; the only behavioral
// difference is that C23 additionally accepts a "0b"/"0B" binary prefix when
// base is 0 or 2. ONNX Runtime only parses decimal/hex config values, so
// forwarding to the pre-C23 functions is safe.
//
// We define the symbols here and forward to the classic glibc functions:
//   - on glibc < 2.38 they resolve the otherwise-undefined references;
//   - on glibc >= 2.38 they merely interpose glibc's own definitions (defining
//     a symbol in the executable that also exists in a shared library is fine
//     — no duplicate-symbol error, our definition wins).
//
// Gated to exactly the configuration that links the prebuilt archive:
// linux-gnu with bundled-embeddings. musl, Windows, and macOS never reference
// __isoc23_* and must not carry the shims.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_long, c_longlong, c_ulong, c_ulonglong, c_void};

// `locale_t` is an opaque pointer type in glibc.
#[allow(non_camel_case_types)]
type locale_t = *mut c_void;

extern "C" {
    fn strtol(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoll(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_longlong;
    fn strtoul(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_ulong;
    fn strtoull(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_ulonglong;
    fn strtoll_l(
        nptr: *const c_char,
        endptr: *mut *mut c_char,
        base: c_int,
        loc: locale_t,
    ) -> c_longlong;
    fn strtoull_l(
        nptr: *const c_char,
        endptr: *mut *mut c_char,
        base: c_int,
        loc: locale_t,
    ) -> c_ulonglong;
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtol(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_long {
    strtol(nptr, endptr, base)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoll(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_longlong {
    strtoll(nptr, endptr, base)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoul(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_ulong {
    strtoul(nptr, endptr, base)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoull(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_ulonglong {
    strtoull(nptr, endptr, base)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoll_l(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
    loc: locale_t,
) -> c_longlong {
    strtoll_l(nptr, endptr, base, loc)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoull_l(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
    loc: locale_t,
) -> c_ulonglong {
    strtoull_l(nptr, endptr, base, loc)
}
