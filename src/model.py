"""
Copyright (c) 2025 hazy

Author: hazy
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
from log import LoggingManager
import logging

# originally log dir
from pathlib import Path
current_dir = Path(__file__).parent.resolve()
log_dir = current_dir / "logs"
print(f"Log directory: {log_dir}")
# config manager
from config import config_instance, LOGS_DIR
print(f"Config LOGS_DIR: {LOGS_DIR}")


log_mgr = LoggingManager(LOGS_DIR / "app.log", level=logging.DEBUG)
logger = log_mgr.get_logger(__name__)
logger.info("app start")
logger.debug("debug message")
logger.warning(f"warning message {LOGS_DIR}", extra={"log_dir": LOGS_DIR})

# 程序退出时
log_mgr.shutdown()
