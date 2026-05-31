"""统一数据源接口: DummySource (模拟数据) + CameraSource (真实摄像头占位)。"""
from abc import ABC, abstractmethod
import numpy as np


class DataSource(ABC):
    @abstractmethod
    def get_input(self) -> dict:
        """返回 engine 推理需要的输入 dict {name: np.ndarray}"""
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        """返回数据源信息，供报告引用"""
        ...


class DummySource(DataSource):
    """随机生成模拟输入，立即可用"""

    def __init__(self, input_name='images', shape=(1, 3, 640, 640), dtype=np.float32):
        self.input_name = input_name
        self.shape = shape
        self.dtype = dtype

    def get_input(self):
        return {self.input_name: np.random.randn(*self.shape).astype(self.dtype)}

    def get_metadata(self):
        return {
            'source': 'dummy',
            'input_name': self.input_name,
            'shape': self.shape,
            'dtype': str(self.dtype),
            'description': '随机生成的正态分布 N(0,1) 数据'
        }


class CameraSource(DataSource):
    """真实摄像头数据源 — 占位实现，等待摄像头接入后二次分析

    接入步骤：
    1. pip install opencv-python (如果未安装)
    2. 取消下方 get_input() 中的注释，替换 DummySource 调用
    3. 根据需要调整 device_id 和图像预处理
    """

    def __init__(self, device_id=0, input_name='images', shape=(1, 3, 640, 640), dtype=np.float32):
        self.device_id = device_id
        self.input_name = input_name
        self.shape = shape
        self.dtype = dtype

    def get_input(self):
        raise NotImplementedError(
            "CameraSource 尚未实现。\n"
            "接入摄像头后，参考以下实现：\n"
            "  import cv2\n"
            "  cap = cv2.VideoCapture(self.device_id)\n"
            "  ret, frame = cap.read()\n"
            "  frame = cv2.resize(frame, (self.shape[3], self.shape[2]))\n"
            "  frame = frame.transpose(2, 0, 1)[None].astype(np.float32) / 255.0\n"
            "  return {self.input_name: frame}"
        )

    def get_metadata(self):
        return {
            'source': 'camera',
            'device_id': self.device_id,
            'input_name': self.input_name,
            'shape': self.shape,
            'dtype': str(self.dtype),
            'status': 'NOT_IMPLEMENTED'
        }
