"""TensorRT engine 封装: 构建、加载和推理

使用 ctypes 调用 CUDA driver API, 无需 PyCUDA 依赖。
"""

import os
import ctypes
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ---- dtype mapping (workaround for numpy 1.24+ removing np.bool) ----
_TRT_NPTYPE = {
    trt.float32: np.float32,
    trt.float16: np.float16,
    trt.int32: np.int32,
    trt.int8: np.int8,
    trt.bool: np.bool_,
}
if hasattr(trt, 'uint8'):
    _TRT_NPTYPE[trt.uint8] = np.uint8

# ---- CUDA driver API (via ctypes) ----
_libcuda = ctypes.CDLL('libcuda.so')

_cu_mem_alloc = _libcuda.cuMemAlloc_v2
_cu_mem_alloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
_cu_mem_alloc.restype = int

_cu_mem_free = _libcuda.cuMemFree_v2
_cu_mem_free.argtypes = [ctypes.c_ulonglong]
_cu_mem_free.restype = int

_cu_memcpy_htod = _libcuda.cuMemcpyHtoD_v2
_cu_memcpy_htod.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]
_cu_memcpy_htod.restype = int

_cu_memcpy_dtoh = _libcuda.cuMemcpyDtoH_v2
_cu_memcpy_dtoh.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]
_cu_memcpy_dtoh.restype = int


def _gpu_alloc(size):
    ptr = ctypes.c_ulonglong()
    err = _cu_mem_alloc(ctypes.byref(ptr), size)
    if err != 0:
        raise RuntimeError(f"cuMemAlloc failed: {err}")
    return ptr.value


def _gpu_free(ptr):
    _cu_mem_free(ctypes.c_ulonglong(ptr))


def _memcpy_htod(gpu_ptr, cpu_arr):
    _cu_memcpy_htod(ctypes.c_ulonglong(gpu_ptr), cpu_arr.ctypes.data, cpu_arr.nbytes)


def _memcpy_dtoh(cpu_arr, gpu_ptr):
    _cu_memcpy_dtoh(cpu_arr.ctypes.data, ctypes.c_ulonglong(gpu_ptr), cpu_arr.nbytes)


class TrtEngine:
    """加载序列化的 TensorRT engine 并执行推理"""

    def __init__(self, engine_path: str):
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        self.buffers = {}
        self._gpu_ptrs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = np.dtype(_TRT_NPTYPE.get(self.engine.get_tensor_dtype(name), np.float32))
            n_elements = int(np.prod(shape))
            size_bytes = n_elements * dtype.itemsize

            gpu_ptr = _gpu_alloc(size_bytes)
            self._gpu_ptrs.append(gpu_ptr)
            cpu_mem = np.empty(n_elements, dtype=dtype)

            self.buffers[name] = {
                'gpu': gpu_ptr, 'cpu': cpu_mem, 'shape': shape, 'dtype': dtype
            }

            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"[TRT] 加载 {engine_path}: {len(self.input_names)} 输入, {len(self.output_names)} 输出")

    def infer(self, input_dict: dict) -> dict:
        """执行推理, 返回 {name: numpy_array}"""
        for name, data in input_dict.items():
            buf = self.buffers[name]
            buf['cpu'][:data.size] = np.ascontiguousarray(data, dtype=buf['dtype']).ravel()
            _memcpy_htod(buf['gpu'], buf['cpu'])

        for name in self.input_names:
            self.context.set_tensor_address(name, self.buffers[name]['gpu'])
        for name in self.output_names:
            self.context.set_tensor_address(name, self.buffers[name]['gpu'])

        self.context.execute_async_v3(0)

        outputs = {}
        for name in self.output_names:
            buf = self.buffers[name]
            _memcpy_dtoh(buf['cpu'], buf['gpu'])
            outputs[name] = buf['cpu'].reshape(buf['shape'])

        return outputs

    def __del__(self):
        for ptr in getattr(self, '_gpu_ptrs', []):
            try:
                _gpu_free(ptr)
            except Exception:
                pass


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 dynamic_shapes: dict = None):
    """从 ONNX 构建 TensorRT engine 并序列化

    dynamic_shapes: {tensor_name: (min_shape, opt_shape, max_shape)} 用于动态输入
    """
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[TRT] parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, (min_s, opt_s, max_s) in dynamic_shapes.items():
            profile.set_shape(name, min_s, opt_s, max_s)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Failed to build engine from {onnx_path}")

    os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(bytes(serialized))

    print(f"[TRT] Engine saved to {engine_path}")
