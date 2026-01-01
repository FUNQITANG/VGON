"""
Copyright (c) 2025 cyy https://github.com/Cyy-Asti

Author: cyy 
Editor: hazy
Created: 2025-12
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# 尝试从统一配置导入，失败则退化到本地环境配置
try:
    from ..config import DEVICE, PROJECT_ROOT
    # print(f"Using device: {DEVICE} (from config)")
    # print(f"PROJECT_ROOT: {PROJECT_ROOT} (from config)")
except (ImportError, ValueError) as e:
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.resolve()

    import logging
    from logging.handlers import RotatingFileHandler
    simple_logger_str  = "%(asctime)s %(levelname)-8s %(message)s"
    extra_logger_str   = "[%(process)d:%(threadName)s]%(name)s @"
    logger_filename     = "VAEClass_fallback"
    logger_size        =100 * 1024 * 1024         # 100MB
    logger_file_path     = PROJECT_ROOT / "logs"
    log_timestamp      = "%Y%m%d_%Hh%M_%S"
    # log_timestamp      = None
    
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(logging.Formatter(
        fmt =simple_logger_str ,
        datefmt="%H:%M"))

    file_handler = logging.NullHandler()
    if log_timestamp is None:
        file_handler = RotatingFileHandler(
            filename=logger_file_path / (logger_filename + ".log"), 
            maxBytes=logger_size,
            backupCount=5,
            encoding="utf-8"
        )   # 5 backup if size exceeds, total up tp 500MB log files
    else:
        from datetime import datetime
        logger_file = logger_file_path / (logger_filename + datetime.now().strftime(log_timestamp) + ".log")
        file_handler = logging.FileHandler(
            filename=logger_file, 
            encoding="utf-8"
        )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt =extra_logger_str + simple_logger_str, 
        datefmt="%Y-%m-%d %H:%M:%S"))
    
    logging.shutdown()      # 单模组运行，不应当再有其他同名logger残留，每次运行都新建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 必须设置 logger 本身的级别，否则默认 WARNING 会过滤掉低级别日志
    # 避免重复添加 handler
    if not logger.handlers:
        logger.addHandler(terminal_handler)
        logger.addHandler(file_handler)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.warning(f"Failed to import configuration from config.py; using local defaults.\n reason: {e}")

logger.info("VAEClass module initialized.")
logger.info(f"Using device: {DEVICE}")
logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"log file path: {logger_file_path}")



class VAEModel(nn.Module):
    def __init__(self, para_dim, z_dim, h_dim, post_sigma, amplify, activation, Device=DEVICE):
        super(VAEModel, self).__init__()
        self.post_sigma = post_sigma
        self.amplify = amplify

        # activation
        # self.activation_type = activation
        self.activate_funcs = dict(
            relu=F.relu,
            tanh=F.tanh,
            sigmoid=F.sigmoid,
            leaky_relu=F.leaky_relu,
            selu=F.selu,
            identity=lambda x: x
        )
        self.activation = self.activate_funcs.get(activation, None)     # 初始化时检查，默认运行时不再改变。
        if self.activation is None:
            logger.error(f"Unknown activate function: {activation}. What available: {list(self.activate_funcs.keys())}")
            raise ValueError(f"Unknown activate function: {activation}")
        
        self.device = Device

        # encoder
        self.e1 = nn.ModuleList([nn.Linear(para_dim, h_dim[0],bias=True)])
        self.e1 += [nn.Linear(h_dim[i-1], h_dim[i],bias=True) for i in range(1, len(h_dim))]
        self.e2 = nn.Linear(h_dim[-1], z_dim,bias=True) # get mean prediction
        self.e3 = nn.Linear(h_dim[-1], z_dim,bias=True) # get mean prediction

        # decoder
        self.d4 = nn.ModuleList([nn.Linear(z_dim, h_dim[-1],bias=True)])
        self.d4 += [nn.Linear(h_dim[-i+1], h_dim[-i],bias=True) for i in range(2, len(h_dim)+1)]
        self.d5 = nn.Linear(h_dim[0], para_dim,bias=True)


    def encoder(self, x):
        h = self.activation(self.e1[0](x))
        for i in range(1, len(self.e1)):
            h = self.activation(self.e1[i](h))
        # get_mean
        mean = self.e2(h)
        # get_variance
        log_var = self.e3(h)  
        return mean, log_var

    def reparameterize(self, mean, log_var, randomness, Display=False):
        if randomness:
            eps = torch.randn(log_var.shape) * self.post_sigma
            logger.debug(f"Reparameterization with randomness.eps {eps} generated.")
            if Display:
                logger.info("Using stochastic mode, randomness applied.")
            # print("=============eps: ================\n", eps)
        else:
            eps = torch.zeros(log_var.shape)
            logger.debug(f"Reparameterization without randomness.'eps' set to zeros with size {eps.size()}.")
            if Display:
                logger.info("Using deterministic mode, no randomness applied.")
            
        std = torch.exp(log_var).pow(0.5) # square root
        eps = eps.to(self.device)
        std = std.to(self.device)
        z = mean + std * eps
        return z

    def decoder(self, z):
        out = self.activation(self.d4[0](z))
        for i in range(1, len(self.d4)):
            out = self.activation(self.d4[i](out))
        out = self.d5(out)
        return out * self.amplify

    def forward(self, x, randomness,Display=False):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var, randomness,Display)
        out = self.decoder(z)
        return out, mean, log_var
    