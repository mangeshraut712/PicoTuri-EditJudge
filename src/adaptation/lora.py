"""
LoRA Adapter Training
Low-Rank Adaptation for domain-specific fine-tuning in PicoTuri-EditJudge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import numpy as np
from pathlib import Path
import json
from transformers import AutoModel, AutoTokenizer
import math
from enum import Enum

logger = logging.getLogger(__name__)

class LoRARank(Enum):
    """Predefined LoRA ranks for different adaptation scenarios"""
    MINIMAL = 4    # Very lightweight adaptation
    SMALL = 8      # Small adaptation for minor domain shifts
    MEDIUM = 16    # Medium adaptation for moderate domain shifts
    LARGE = 32     # Large adaptation for significant domain shifts
    XLARGE = 64    # Extra large adaptation for major domain shifts

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "value", "dense", "intermediate"
    ])
    bias: str = "none"  # "none", "all", "lora_only"
    fan_in_fan_out: bool = False
    init_lora_weights: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        fan_in_fan_out: bool = False
    ):
        """
        Initialize LoRA layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor
            dropout: Dropout probability
            fan_in_fan_out: Whether to transpose weight matrices
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.fan_in_fan_out = fan_in_fan_out
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
        
        # Disable gradient for original weights (handled by parent module)
        self.weight = None  # Will be set by parent
        self.bias = None    # Will be set by parent
    
    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        if self.weight is None:
            raise RuntimeError("LoRA layer weight not set")
        
        # Original linear transformation
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA adaptation
        if self.rank > 0:
            lora_result = (
                self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            ) * self.scaling
            result = result + lora_result
        
        return result
    
    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get LoRA parameters only"""
        return [self.lora_A, self.lora_B]
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA state dict for saving"""
        return {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B
        }
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA state dict"""
        self.lora_A.data = state_dict["lora_A"]
        self.lora_B.data = state_dict["lora_B"]

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_config: Optional[LoRAConfig] = None
    ):
        """
        Initialize LoRA linear layer
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            lora_config: LoRA configuration
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adaptation
        if lora_config is not None:
            self.lora = LoRALayer(
                in_features=in_features,
                out_features=out_features,
                rank=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                fan_in_fan_out=lora_config.fan_in_fan_out
            )
            self.lora.weight = self.linear.weight
            self.lora.bias = self.linear.bias
        else:
            self.lora = None
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.lora is not None:
            return self.lora(x)
        else:
            return self.linear(x)
    
    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get LoRA parameters"""
        if self.lora is not None:
            return self.lora.get_lora_parameters()
        return []
    
    def enable_adapter_layers(self):
        """Enable LoRA adapter layers"""
        if self.lora is not None:
            for param in self.get_lora_parameters():
                param.requires_grad = True
    
    def disable_adapter_layers(self):
        """Disable LoRA adapter layers"""
        for param in self.get_lora_parameters():
            param.requires_grad = False

class LoRAAdapter:
    """
    LoRA adapter for transformer models
    """
    
    def __init__(
        self,
        base_model_name: str,
        lora_config: LoRAConfig,
        device: str = "auto"
    ):
        """
        Initialize LoRA adapter
        
        Args:
            base_model_name: Name of the base model
            lora_config: LoRA configuration
            device: Device to run on
        """
        self.base_model_name = base_model_name
        self.lora_config = lora_config
        self.device = self._get_device(device)
        
        # Load base model and tokenizer
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Apply LoRA to target modules
        self._apply_lora()
        
        # Move to device
        self.base_model.to(self.device)
        
        logger.info(f"LoRA adapter initialized for {base_model_name}")
    
    def _get_device(self, device: str) -> str:
        """Get appropriate device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _apply_lora(self):
        """Apply LoRA to target modules"""
        target_modules = self.lora_config.target_modules
        
        for name, module in self.base_model.named_modules():
            # Check if module name contains any target module
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA linear layer
                    parent_module = self.base_model.get_submodule(name.rsplit('.', 1)[0])
                    child_name = name.rsplit('.', 1)[1]
                    
                    lora_linear = LoRALinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        lora_config=self.lora_config
                    )
                    
                    # Copy weights
                    lora_linear.linear.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_linear.linear.bias.data = module.bias.data.clone()
                    
                    # Replace module
                    setattr(parent_module, child_name, lora_linear)
                    
                    logger.debug(f"Applied LoRA to {name}")
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable (LoRA) parameters"""
        trainable_params = []
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                trainable_params.extend(module.get_lora_parameters())
        return trainable_params
    
    def enable_training(self):
        """Enable training mode for LoRA parameters"""
        self.base_model.train()
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.enable_adapter_layers()
    
    def disable_training(self):
        """Disable training mode"""
        self.base_model.eval()
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.disable_adapter_layers()
    
    def save_adapter(self, save_path: str):
        """
        Save LoRA adapter weights and configuration
        
        Args:
            save_path: Path to save adapter
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        lora_state_dict = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALinear) and module.lora is not None:
                lora_state_dict[name] = module.lora.get_lora_state_dict()
        
        torch.save(lora_state_dict, save_path / "adapter_weights.pt")
        
        # Save configuration
        config = {
            "base_model_name": self.base_model_name,
            "lora_config": {
                "rank": self.lora_config.rank,
                "alpha": self.lora_config.alpha,
                "dropout": self.lora_config.dropout,
                "target_modules": self.lora_config.target_modules,
                "bias": self.lora_config.bias,
                "fan_in_fan_out": self.lora_config.fan_in_fan_out,
                "init_lora_weights": self.lora_config.init_lora_weights
            }
        }
        
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"LoRA adapter saved to {save_path}")
    
    @classmethod
    def load_adapter(
        cls,
        adapter_path: str,
        device: str = "auto"
    ) -> "LoRAAdapter":
        """
        Load LoRA adapter from saved checkpoint
        
        Args:
            adapter_path: Path to saved adapter
            device: Device to run on
            
        Returns:
            Loaded LoRA adapter
        """
        adapter_path = Path(adapter_path)
        
        # Load configuration
        with open(adapter_path / "adapter_config.json", "r") as f:
            config = json.load(f)
        
        lora_config = LoRAConfig(**config["lora_config"])
        
        # Create adapter
        adapter = cls(
            base_model_name=config["base_model_name"],
            lora_config=lora_config,
            device=device
        )
        
        # Load LoRA weights
        lora_state_dict = torch.load(adapter_path / "adapter_weights.pt", map_location=device)
        
        for name, module in adapter.base_model.named_modules():
            if isinstance(module, LoRALinear) and module.lora is not None:
                if name in lora_state_dict:
                    module.lora.load_lora_state_dict(lora_state_dict[name])
        
        logger.info(f"LoRA adapter loaded from {adapter_path}")
        return adapter
    
    def merge_adapter(self):
        """
        Merge LoRA weights into the base model
        This makes the adaptation permanent and removes the LoRA overhead
        """
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALinear) and module.lora is not None:
                # Merge LoRA weights
                with torch.no_grad():
                    lora_weight = (
                        module.lora.lora_B @ module.lora.lora_A
                    ) * module.lora.scaling
                    
                    # Transpose if needed
                    if module.lora.fan_in_fan_out:
                        lora_weight = lora_weight.T
                    
                    # Add to original weights
                    module.linear.weight.data += lora_weight
                
                # Replace with regular linear layer
                parent_module = self.base_model.get_submodule(name.rsplit('.', 1)[0])
                child_name = name.rsplit('.', 1)[1]
                setattr(parent_module, child_name, module.linear)
        
        logger.info("LoRA adapter merged into base model")

class LoRATrainer:
    """
    Trainer for LoRA adapters
    """
    
    def __init__(
        self,
        adapter: LoRAAdapter,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        save_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10
    ):
        """
        Initialize LoRA trainer
        
        Args:
            adapter: LoRA adapter to train
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of training steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
        """
        self.adapter = adapter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        # Setup optimizer
        trainable_params = adapter.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info("LoRA trainer initialized")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(self.max_steps - current_step) / float(max(1, self.max_steps - self.warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: callable
    ) -> float:
        """
        Perform one training step
        
        Args:
            batch: Training batch
            loss_fn: Loss function
            
        Returns:
            Training loss
        """
        self.adapter.enable_training()
        
        # Forward pass
        outputs = self.adapter.base_model(**batch)
        loss = loss_fn(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.adapter.get_trainable_parameters(),
            max_norm=1.0
        )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item()
    
    def evaluate(
        self,
        eval_dataloader: torch.utils.data.DataLoader,
        loss_fn: callable
    ) -> Dict[str, float]:
        """
        Evaluate the adapter
        
        Args:
            eval_dataloader: Evaluation data loader
            loss_fn: Loss function
            
        Returns:
            Evaluation metrics
        """
        self.adapter.disable_training()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.adapter.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.adapter.base_model(**batch)
                loss = loss_fn(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "eval_loss": avg_loss,
            "perplexity": math.exp(avg_loss) if avg_loss < 20 else float('inf')
        }
    
    def save_checkpoint(self, save_path: str, step: int):
        """
        Save training checkpoint
        
        Args:
            save_path: Path to save checkpoint
            step: Current training step
        """
        checkpoint_path = Path(save_path) / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save adapter
        self.adapter.save_adapter(checkpoint_path)
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save scheduler state
        torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay
        }
        
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")

# Utility functions
def create_lora_config(
    rank: Union[int, LoRARank] = LoRARank.MEDIUM,
    alpha: Optional[float] = None,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> LoRAConfig:
    """
    Create LoRA configuration with sensible defaults
    
    Args:
        rank: LoRA rank (int or predefined rank)
        alpha: Scaling factor (defaults to 2 * rank)
        dropout: Dropout probability
        target_modules: Target modules for adaptation
        
    Returns:
        LoRA configuration
    """
    if isinstance(rank, LoRARank):
        rank_value = rank.value
    else:
        rank_value = rank
    
    if alpha is None:
        alpha = 2 * rank_value
    
    if target_modules is None:
        target_modules = ["query", "value", "dense", "intermediate"]
    
    return LoRAConfig(
        rank=rank_value,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules
    )

def estimate_lora_parameters(
    base_model: nn.Module,
    lora_config: LoRAConfig
) -> Dict[str, int]:
    """
    Estimate number of parameters added by LoRA
    
    Args:
        base_model: Base model
        lora_config: LoRA configuration
        
    Returns:
        Parameter count information
    """
    base_params = sum(p.numel() for p in base_model.parameters())
    
    # Estimate LoRA parameters
    lora_params = 0
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in lora_config.target_modules):
                # LoRA adds: (in_features * rank) + (rank * out_features)
                lora_params += module.in_features * lora_config.rank
                lora_params += lora_config.rank * module.out_features
    
    return {
        "base_parameters": base_params,
        "lora_parameters": lora_params,
        "total_parameters": base_params + lora_params,
        "lora_percentage": (lora_params / base_params) * 100
    }

# Test function
def test_lora_adapter():
    """Test LoRA adapter functionality"""
    print("Testing LoRA Adapter...")
    
    # Create LoRA config
    lora_config = create_lora_config(
        rank=LoRARank.SMALL,
        alpha=16.0,
        dropout=0.1
    )
    
    print(f"LoRA config: rank={lora_config.rank}, alpha={lora_config.alpha}")
    
    # Create adapter (using a small model for testing)
    adapter = LoRAAdapter(
        base_model_name="distilbert-base-uncased",
        lora_config=lora_config,
        device="cpu"
    )
    
    # Check trainable parameters
    trainable_params = adapter.get_trainable_parameters()
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Test forward pass
    adapter.enable_training()
    
    # Create dummy input
    inputs = adapter.tokenizer(
        "This is a test sentence for LoRA adaptation.",
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = adapter.base_model(**inputs)
        print(f"Output shape: {outputs.last_hidden_state.shape}")
    
    # Test saving and loading
    adapter.save_adapter("/tmp/test_lora_adapter")
    
    loaded_adapter = LoRAAdapter.load_adapter(
        "/tmp/test_lora_adapter",
        device="cpu"
    )
    
    print("LoRA adapter test completed!")

if __name__ == "__main__":
    test_lora_adapter()
