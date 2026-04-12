"""NepTTS-Eval: Evaluation toolkit for Nepali TTS systems."""

__version__ = "0.1.0"

from .synthesize import benchmark, generate_benchmark_audio

__all__ = ["benchmark", "generate_benchmark_audio", "__version__"]
