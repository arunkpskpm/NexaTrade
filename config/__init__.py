"""
NexaTrade — Config package.
Exposes the unified settings object and config loader.
"""

from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]