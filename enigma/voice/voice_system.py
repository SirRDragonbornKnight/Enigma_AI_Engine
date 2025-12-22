"""
Enhanced Voice System for Enigma

Supports multiple TTS backends with quality levels:
  1. Piper (best quality, offline, fast)
  2. pyttsx3 (decent, offline)
  3. espeak (robotic but reliable)

USAGE:
    from enigma.voice.voice_system import VoiceSystem
    
    voice = VoiceSystem()
    voice.speak("Hello, I am Enigma.")
    
    # Or with specific voice
    voice.speak("Hello", voice_id="en_US-lessac-medium")
    
    # List available voices
    voice.list_voices()
"""

import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

# Check available backends
HAVE_PYTTSX3 = False
HAVE_PIPER = False

try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except ImportError:
    pass

# Check for piper binary
PIPER_PATH = os.environ.get("PIPER_PATH", "piper")
try:
    result = subprocess.run([PIPER_PATH, "--help"], capture_output=True, timeout=5)
    HAVE_PIPER = True
except:
    pass


class VoiceSystem:
    """
    Unified voice system with multiple backends.
    """
    
    def __init__(self, backend: str = "auto", voice_id: Optional[str] = None):
        """
        Args:
            backend: "piper", "pyttsx3", "espeak", or "auto" (best available)
            voice_id: Voice identifier (backend-specific)
        """
        self.backend = self._select_backend(backend)
        self.voice_id = voice_id
        self._pyttsx3_engine = None
        
        print(f"Voice system initialized: {self.backend}")
    
    def _select_backend(self, requested: str) -> str:
        """Select the best available backend."""
        if requested == "auto":
            if HAVE_PIPER:
                return "piper"
            elif HAVE_PYTTSX3:
                return "pyttsx3"
            else:
                return "espeak"
        return requested
    
    def speak(self, text: str, voice_id: Optional[str] = None, rate: int = 150):
        """
        Speak text aloud.
        
        Args:
            text: Text to speak
            voice_id: Override default voice
            rate: Speech rate (words per minute)
        """
        if not text:
            return
        
        voice = voice_id or self.voice_id
        
        if self.backend == "piper":
            self._speak_piper(text, voice)
        elif self.backend == "pyttsx3":
            self._speak_pyttsx3(text, rate)
        else:
            self._speak_espeak(text, rate)
    
    def _speak_pyttsx3(self, text: str, rate: int):
        """Use pyttsx3 for TTS."""
        try:
            if self._pyttsx3_engine is None:
                self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty('rate', rate)
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as e:
            print(f"pyttsx3 failed: {e}, falling back to espeak")
            self._speak_espeak(text, rate)
    
    def _speak_espeak(self, text: str, rate: int):
        """Use espeak for TTS (fallback)."""
        try:
            # Sanitize text for shell
            safe_text = text.replace('"', '\\"').replace("'", "\\'")
            speed = max(80, min(450, rate))  # espeak range
            os.system(f'espeak -s {speed} "{safe_text}" 2>/dev/null')
        except Exception as e:
            print(f"espeak failed: {e}")
    
    def _speak_piper(self, text: str, voice: Optional[str]):
        """Use Piper for high-quality TTS."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            
            cmd = [PIPER_PATH, "--output_file", wav_path]
            if voice:
                cmd.extend(["--model", voice])
            
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.communicate(input=text.encode())
            
            # Play the wav file
            if os.path.exists(wav_path):
                os.system(f"aplay {wav_path} 2>/dev/null || paplay {wav_path} 2>/dev/null")
                os.unlink(wav_path)
        except Exception as e:
            print(f"Piper failed: {e}, falling back")
            self._speak_espeak(text, 150)
    
    def list_voices(self) -> List[str]:
        """List available voices for current backend."""
        if self.backend == "pyttsx3" and HAVE_PYTTSX3:
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                return [v.id for v in voices]
            except:
                return []
        elif self.backend == "espeak":
            # List espeak voices
            try:
                result = subprocess.run(["espeak", "--voices"], capture_output=True, text=True)
                return result.stdout.strip().split('\n')[1:]  # Skip header
            except:
                return []
        return []
    
    def set_voice(self, voice_id: str):
        """Set the voice to use."""
        self.voice_id = voice_id
        if self.backend == "pyttsx3" and self._pyttsx3_engine:
            try:
                self._pyttsx3_engine.setProperty('voice', voice_id)
            except:
                pass


# Simple function interface (backwards compatible)
_default_voice = None

def speak(text: str, rate: int = 150):
    """Simple speak function."""
    global _default_voice
    if _default_voice is None:
        _default_voice = VoiceSystem()
    _default_voice.speak(text, rate=rate)


def get_voice_system() -> VoiceSystem:
    """Get the voice system instance."""
    global _default_voice
    if _default_voice is None:
        _default_voice = VoiceSystem()
    return _default_voice


if __name__ == "__main__":
    print("Testing Voice System...")
    print(f"Available backends: pyttsx3={HAVE_PYTTSX3}, piper={HAVE_PIPER}, espeak=True")
    
    voice = VoiceSystem()
    voice.speak("Hello, I am Enigma. The voice system is working correctly.")
    
    print("\nAvailable voices:")
    for v in voice.list_voices()[:5]:  # First 5
        print(f"  {v}")
