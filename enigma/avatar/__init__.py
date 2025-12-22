# avatar package
"""
Enigma Avatar System

Provides a controllable avatar that can:
  - Display on screen (on/off toggle, default OFF)
  - Move around the desktop
  - Express emotions
  - "Interact" with windows and files
  - Control a 3D model (when renderer implemented)

USAGE:
    from enigma.avatar import get_avatar, toggle_avatar
    
    avatar = get_avatar()
    avatar.enable()  # Turn on (default is off!)
    avatar.move_to(500, 300)
    avatar.speak("Hello!")
    avatar.interact_with_window("My Document")
    avatar.disable()
"""

from .controller import (
    AvatarController,
    AvatarConfig,
    AvatarState,
    AvatarPosition,
    get_avatar,
    enable_avatar,
    disable_avatar,
    toggle_avatar,
)

__all__ = [
    "AvatarController",
    "AvatarConfig",
    "AvatarState",
    "AvatarPosition",
    "get_avatar",
    "enable_avatar",
    "disable_avatar",
    "toggle_avatar",
]
