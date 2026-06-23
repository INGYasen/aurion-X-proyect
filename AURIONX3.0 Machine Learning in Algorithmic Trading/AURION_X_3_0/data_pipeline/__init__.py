"""AURION X 4.1 — pipeline de datos históricos para XAU Knowledge Engine."""

from .xau_downloader import (
    XAUDownloader,
    XAUDownloaderConfig,
    TimeframeSpec,
    SUPPORTED_TIMEFRAMES,
    download_xauusd_history,
)

__all__ = [
    "XAUDownloader",
    "XAUDownloaderConfig",
    "TimeframeSpec",
    "SUPPORTED_TIMEFRAMES",
    "download_xauusd_history",
]
