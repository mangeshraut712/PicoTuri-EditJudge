"""
Adaptation Module
Custom domain adaptation with LoRA and calibration for PicoTuri-EditJudge
"""

from .lora import (
    LoRAAdapter,
    LoRALayer,
    LoRALinear,
    LoRATrainer,
    LoRAConfig,
    LoRARank,
    create_lora_config,
    estimate_lora_parameters
)

from .calibration import (
    DomainAdapter,
    BinaryCalibrator,
    DriftDetector,
    TemperatureScaling,
    PlattScaling,
    CalibrationMethod,
    DriftDetectionMethod,
    CalibrationConfig,
    DriftDetectionConfig,
    CalibrationMetrics,
    DriftMetrics
)

__all__ = [
    # LoRA components
    "LoRAAdapter",
    "LoRALayer",
    "LoRALinear",
    "LoRATrainer",
    "LoRAConfig",
    "LoRARank",
    "create_lora_config",
    "estimate_lora_parameters",

    # Calibration components
    "DomainAdapter",
    "BinaryCalibrator",
    "DriftDetector",
    "TemperatureScaling",
    "PlattScaling",
    "CalibrationMethod",
    "DriftDetectionMethod",
    "CalibrationConfig",
    "DriftDetectionConfig",
    "CalibrationMetrics",
    "DriftMetrics",
]

# Version info
__version__ = "0.2.0"
__author__ = "PicoTuri Team"
__description__ = "Custom domain adaptation with LoRA and calibration for EditJudge"
