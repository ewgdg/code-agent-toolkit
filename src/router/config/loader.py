import os
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Type, cast, Any

import structlog
import yaml
from pydantic import ValidationError
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .schema import Config

logger = structlog.get_logger(__name__)


class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, config_loader: "ConfigLoader"):
        self.config_loader = config_loader

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if event.src_path == str(self.config_loader.config_path):
            logger.info("Config file changed, reloading", path=event.src_path)
            self.config_loader.reload()


class ConfigLoader:
    def __init__(
        self, config_path: Path | None = None, enable_hot_reload: bool = False
    ):
        self.config_path = config_path or Path("router.yaml")
        self.enable_hot_reload = enable_hot_reload
        self._config: Config | None = None
        self._last_modified: float | None = None
        self._observer: BaseObserver | None = None

        self.load()

        if enable_hot_reload:
            self._setup_hot_reload()

    def load(self) -> Config:
        """Load configuration from file or return defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    data = yaml.safe_load(f)
                    if data is None:
                        data = {}

                self._config = Config(**data)
                self._last_modified = os.path.getmtime(self.config_path)
                logger.info("Config loaded successfully", path=str(self.config_path))
            else:
                self._config = Config()
                logger.info(
                    "Config file not found, using defaults", path=str(self.config_path)
                )

        except (yaml.YAMLError, ValidationError) as e:
            logger.error("Failed to load config, using defaults", error=str(e))
            self._config = Config()
        except Exception as e:
            logger.error(
                "Unexpected error loading config, using defaults", error=str(e)
            )
            self._config = Config()

        return self._config

    def reload(self) -> Config:
        """Reload configuration if file has changed."""
        if not self.config_path.exists():
            return self._config or Config()

        current_mtime = os.path.getmtime(self.config_path)
        if self._last_modified is None or current_mtime > self._last_modified:
            return self.load()

        return self._config or Config()

    def get_config(self) -> Config:
        """Get current configuration."""
        if self._config is None:
            return self.load()
        return self._config

    def _setup_hot_reload(self) -> None:
        """Setup file watching for hot reload."""
        if not self.config_path.exists():
            return

        event_handler = ConfigReloadHandler(self)
        self._observer = Observer()
        self._observer.schedule(
            event_handler, str(self.config_path.parent), recursive=False
        )
        self._observer.start()
        logger.info("Hot reload enabled for config file", path=str(self.config_path))

    def stop_hot_reload(self) -> None:
        """Stop file watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Hot reload stopped")

    def __enter__(self) -> "ConfigLoader":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop_hot_reload()
