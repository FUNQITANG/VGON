"""
Copyright (c) 2025 hazy

Author: hazy
Created: 2025-12-31
Last Modified: 2025-12-31

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch
from pathlib import Path


class Config:
    """全局配置管理类 - 单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 读取同路径 .env（优先级最高），其次系统环境变量，最后默认值
        self._env_file = self._load_env_file()
        
        # ==================== 路径配置 ====================
        self._init_root()
        # 常用目录
        self.SRC_DIR = self.PROJECT_ROOT / 'src'
        self.LOGS_DIR = self.PROJECT_ROOT / 'logs'
        self.RESULTS_DIR = self.PROJECT_ROOT / 'results'
        
        # 创建必要目录
        self.LOGS_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        
        # ==================== GPU设备配置 ====================
        self._init_device()

        
        # ==================== 日志配置 ====================
        self.LOG_LEVEL = self._get_env('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = self._get_env(
            'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # ==================== 模型配置 ====================
        self.DEFAULT_DTYPE = torch.float32
        seed_raw = self._get_env('RANDOM_SEED', '42')
        self.SEED = int(seed_raw)
    
    def _init_root(self):
        """项目根目录：可通过 .env / 环境变量重写"""
        root_override = self._get_env('PROJECT_ROOT', None)
        if root_override:
            self.PROJECT_ROOT = Path(root_override).expanduser().resolve()
            return
        level_raw = self._get_env('PROJECT_ROOT_LEVEL', '2')
        try:
            level = max(1, int(level_raw))
        except ValueError:
            level = 2
        root = Path(__file__).resolve()
        for _ in range(level):
            root = root.parent
        self.PROJECT_ROOT = root

    def _init_device(self):
        """优先从 .env/环境变量读取设备，否则自动检测"""
        device_str = self._get_env('TORCH_DEVICE', 'auto')
        if device_str == 'auto':
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.DEVICE = torch.device(device_str)

        self.CUDA_AVAILABLE = torch.cuda.is_available()
        if self.CUDA_AVAILABLE:
            self.CUDA_DEVICE_COUNT = torch.cuda.device_count()
            self.CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
        else:
            self.CUDA_DEVICE_COUNT = 0
            self.CUDA_DEVICE_NAME = None

    def _load_env_file(self):
        env_path = Path(__file__).with_name('.env')
        data = {}
        if env_path.exists():
            for line in env_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                data[key.strip()] = val.strip().strip('"\'')
        return data

    def _get_env(self, key, default):
        if key in self._env_file:
            return self._env_file[key]
        return os.environ.get(key, default)
        
    def get_log_path(self, filename):
        """获取日志文件的绝对路径"""
        return self.LOGS_DIR / filename
    
    def get_result_path(self, filename):
        """获取结果文件的绝对路径"""
        return self.RESULTS_DIR / filename
    
    def print_config(self):
        """打印当前配置"""
        print("=" * 60)
        print("配置信息:")
        print(f"  项目根目录: {self.PROJECT_ROOT}")
        print(f"  日志目录: {self.LOGS_DIR}")
        print(f"  结果目录: {self.RESULTS_DIR}")
        print(f"  设备: {self.DEVICE}")
        print(f"  CUDA可用: {self.CUDA_AVAILABLE}")
        if self.CUDA_AVAILABLE:
            print(f"  CUDA设备数: {self.CUDA_DEVICE_COUNT}")
            print(f"  CUDA设备名: {self.CUDA_DEVICE_NAME}")
        print(f"  随机种子: {self.SEED}")
        print("=" * 60)


# 创建全局配置实例
config_instance = Config()


# 便捷访问
DEVICE = config_instance.DEVICE
PROJECT_ROOT = config_instance.PROJECT_ROOT
LOGS_DIR = config_instance.LOGS_DIR
RESULTS_DIR = config_instance.RESULTS_DIR