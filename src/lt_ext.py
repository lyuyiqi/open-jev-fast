import os
from torch.utils.cpp_extension import load_inline
_HERE = os.path.dirname(os.path.abspath(__file__))
_C = os.environ["CUDA_HOME"]  # CUDA root with include/ and lib/libcublasLt.so (pip CUDA 13: site-packages/nvidia/cu13)
def load():
    name = os.environ.get("OJ_LT_NAME", "ojlt"); bdir = os.path.join(_HERE, "build_" + name); os.makedirs(bdir, exist_ok=True)
    return load_inline(name=name, cpp_sources=[open(os.path.join(_HERE, "lt.cpp")).read()], functions=["lt_setup", "lt_matmul", "lt2_plan", "lt2_search", "lt2_run", "lt2_info", "lt2_raw", "lt2_add_raw"],
                       with_cuda=True, extra_cflags=["-O3", f"-I{_C}/include"],
                       extra_ldflags=[f"-L{_C}/lib", "-lcublasLt", f"-Wl,-rpath,{_C}/lib"], build_directory=bdir)
