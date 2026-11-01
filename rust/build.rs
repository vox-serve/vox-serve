//! Compiles the CUDA sources with nvcc into one static library:
//! - csrc/fi_plan.cu: C ABI around FlashInfer's host-side prefill planner, against the headers vendored
//!   in third_party/flashinfer (or `FLASHINFER_INCLUDE`);
//! - csrc/sampling.cu: the graph-capturable sampling kernels, launched by the Python programs through
//!   the `vox_serve._rt` extension.
//!
//! libcudart is linked dynamically by soname (libcudart.so.12): torch loads the same library, so the
//! runtime shares torch's CUDA runtime and graph handles / streams are interchangeable. The extension
//! module finds it next to torch in site-packages.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Default SM architectures of the sampling kernels (those nvcc supports), plus PTX for the newest.
const CUDA_ARCHS: &[u32] = &[80, 86, 89, 90, 100, 120];

fn nvcc(cuda_home: &str, args: &[&str]) {
    let status = Command::new(format!("{cuda_home}/bin/nvcc")).args(args).status().expect("failed to run nvcc");
    assert!(status.success(), "nvcc {args:?} failed");
}

/// `VOX_CUDA_ARCHS` (e.g. `90` or `80,90`) or the default list, keeping the archs this nvcc knows.
fn cuda_archs(cuda_home: &str) -> Vec<u32> {
    let out =
        Command::new(format!("{cuda_home}/bin/nvcc")).arg("--list-gpu-arch").output().expect("failed to run nvcc");
    let known: Vec<u32> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("compute_")?.parse().ok())
        .collect();
    let wanted: Vec<u32> = match env::var("VOX_CUDA_ARCHS") {
        Ok(s) => s.split(',').map(|a| a.trim().parse().expect("VOX_CUDA_ARCHS: comma-separated numbers")).collect(),
        Err(_) => CUDA_ARCHS.to_vec(),
    };
    let archs: Vec<u32> = wanted.into_iter().filter(|a| known.contains(a)).collect();
    assert!(!archs.is_empty(), "no CUDA arch of the list is supported by this nvcc");
    archs
}

fn main() {
    for f in ["csrc/fi_plan.cu", "csrc/sampling.cu", "third_party/flashinfer"] {
        println!("cargo:rerun-if-changed={f}");
    }
    for v in ["CUDA_HOME", "VOX_CUDA_ARCH", "VOX_CUDA_ARCHS", "FLASHINFER_INCLUDE"] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let fi_include = env::var("FLASHINFER_INCLUDE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("third_party/flashinfer/include"));
    // The FlashInfer version the planner is built against, checked against the installed package.
    let fi_version = std::fs::read_to_string(fi_include.join("../version.txt")).unwrap_or_default();
    println!("cargo:rustc-env=VOX_FLASHINFER_VERSION={}", fi_version.trim());

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-12".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = |name: &str| out_dir.join(name).to_string_lossy().to_string();
    let common = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC"];

    // The planner only runs host code; the arch just has to satisfy FlashInfer's kernel declarations.
    let fi_arch = env::var("VOX_CUDA_ARCH").unwrap_or_else(|_| "sm_90".to_string());
    let fi_inc = fi_include.to_string_lossy().to_string();
    let mut args = vec!["-c", "csrc/fi_plan.cu", "-o"];
    let fi_obj = obj("fi_plan.o");
    let fi_arch_flag = format!("-arch={fi_arch}");
    args.extend([fi_obj.as_str(), fi_arch_flag.as_str(), "-I", fi_inc.as_str()]);
    args.extend(common);
    nvcc(&cuda_home, &args);

    let archs = cuda_archs(&cuda_home);
    let newest = *archs.iter().max().unwrap();
    let mut gencode: Vec<String> = archs.iter().map(|a| format!("-gencode=arch=compute_{a},code=sm_{a}")).collect();
    gencode.push(format!("-gencode=arch=compute_{newest},code=compute_{newest}"));
    let sm_obj = obj("sampling.o");
    let mut args = vec!["-c", "csrc/sampling.cu", "-o", sm_obj.as_str(), "--use_fast_math", "--threads=0"];
    args.extend(common);
    args.extend(gencode.iter().map(String::as_str));
    nvcc(&cuda_home, &args);

    let lib = out_dir.join("libvoxcuda.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar").arg("crs").arg(&lib).args([&fi_obj, &sm_obj]).status().unwrap();
    assert!(status.success());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=voxcuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Link time: the toolkit's libcudart.so.12. Run time: whichever copy the process loads first.
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-arg=-l:libcudart.so.12");
    // Extension module (site-packages/vox_serve/_rt*.so): the nvidia-cuda-runtime wheel's copy, or the
    // one the torch wheel bundles (aarch64 CUDA builds); importing torch first loads the same file.
    println!("cargo:rustc-link-arg-cdylib=-Wl,-rpath,$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../torch/lib");
}
