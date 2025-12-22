# tools package
"""
Enigma Tools - Capabilities for the AI

Available tools:
  - web_search: Search the internet
  - fetch_webpage: Get content from a URL
  - read_file: Read a file
  - write_file: Write to a file
  - list_directory: List files in a directory
  - move_file: Move/rename a file
  - delete_file: Delete a file
  - read_document: Read books, PDFs, etc.
  - extract_text: Extract text from any file
  - run_command: Execute shell commands
  - screenshot: Take a screenshot
  - get_system_info: Get system information

USAGE:
    from enigma.tools import ToolRegistry, execute_tool
    
    # Quick execute
    result = execute_tool("web_search", query="python tutorials")
    
    # Or use registry
    tools = ToolRegistry()
    result = tools.execute("read_file", path="README.md")
"""

from .tool_registry import Tool, ToolRegistry, get_registry, execute_tool
from .vision import ScreenCapture, ScreenVision, get_screen_vision, ScreenVisionTool, FindOnScreenTool

__all__ = [
    "Tool",
    "ToolRegistry", 
    "get_registry",
    "execute_tool",
    # Vision
    "ScreenCapture",
    "ScreenVision",
    "get_screen_vision",
    "ScreenVisionTool",
    "FindOnScreenTool",
]
