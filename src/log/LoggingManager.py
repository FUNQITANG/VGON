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

import logging
import sys
from pathlib import Path
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import Queue
from threading import RLock


class LoggingManager:
    """
    - get_logger(name): 给不同模块返回不同名字的 logger
    - 异步写入：业务线程 -> QueueHandler 入队；后台 QueueListener 统一写文件/控制台
    """

    def __init__(
        self,
        log_file: str | Path,
        level: int = logging.INFO,
        fmt: str | None = None,
        datefmt: str | None = "%Y-%m-%d %H:%M:%S",
        console: bool = True,
        max_bytes: int = 50 * 1024 * 1024,  # 50MB
        backup_count: int = 10,
    ):
        self._lock = RLock()

        self.level = level
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.fmt = fmt or "%(levelname)s[%(process)d:%(threadName)s]%(name)s(%(asctime)s) say %(message)s"
        self.datefmt = datefmt
        self.formatter = logging.Formatter(self.fmt, datefmt=self.datefmt)

        # 队列：业务线程写入这里
        self._queue: Queue = Queue(-1)

        # 后台真正输出的 handlers（写文件/写控制台）
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)

        handlers = [file_handler]

        if console:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(self.level)
            stream_handler.setFormatter(self.formatter)
            handlers.append(stream_handler)

        # listener：后台线程从队列取日志，交给 handlers 写出
        self._listener = QueueListener(self._queue, *handlers, respect_handler_level=True)
        self._listener.start()

        # 所有 logger 共享同一个 QueueHandler（避免每个 logger 都起一套 handler）
        self._queue_handler = QueueHandler(self._queue)
        self._queue_handler.setLevel(self.level)

        # 用于保证同名 logger 只初始化一次
        self._initialized_logger_names: set[str] = set()

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """
        name：通常传 __name__，以区分模块
        """
        logger = logging.getLogger(name or __name__)

        # 注意：logger 默认 level=WARNING；这里显式设定
        logger.setLevel(self.level)
        logger.propagate = False  # 防止传播到 root 造成重复输出

        with self._lock:
            if logger.name not in self._initialized_logger_names:
                # 避免重复加 handler
                if not any(isinstance(h, QueueHandler) for h in logger.handlers):
                    logger.addHandler(self._queue_handler)
                self._initialized_logger_names.add(logger.name)

        return logger

    def shutdown(self):
        """
        程序退出前调用，确保队列日志写完并关闭文件句柄
        """
        with self._lock:
            self._listener.stop()
            # 额外清理：把 QueueHandler 从各 logger 移除并关闭（可选）
            for name in list(self._initialized_logger_names):
                lg = logging.getLogger(name)
                for h in list(lg.handlers):
                    if isinstance(h, QueueHandler):
                        lg.removeHandler(h)
                        try:
                            h.close()
                        except Exception:
                            pass
            self._initialized_logger_names.clear()


import sys
import logging
from pathlib import Path


class Logger:

    def __init__(self, log_path, level=logging.INFO, logger_name=None, fmt=None):
        fmt = fmt or '%(asctime)s(%(name)s) %(message)s'
        self.logger = logging.getLogger(logger_name or __name__)
        self.logger.setLevel(level)

        formatter = logging.Formatter(fmt)

        log_path = Path(log_path)
        if log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        self.handler = (file_handler, stream_handler)

    def add_handler(self):
        for handler in self.handler:
            if handler not in self.logger.handlers:
                self.logger.addHandler(handler)

    def remove_handler(self):
        for handler in self.handler:
            if handler in self.logger.handlers:
                self.logger.removeHandler(handler)
