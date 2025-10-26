#!/usr/bin/env python3
"""
SFT Training Script for Pico-Banana-400K
Demonstrates Phase 4: Train incrementally (SFT → DPO → Multi-turn)

Focuses on initial SFT (Supervised Fine-tuning) phase to establish
basic editing capabilities before moving to preference learning.
"""

from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only imports for better IDE support
    import torch  # type: ignore[import-untyped]
    import torch.nn as nn  # type: ignore[import-untyped]
    from torch.utils.data import Dataset, DataLoader  # type: ignore[import-untyped]

# Runtime imports
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
from torch.utils.data import Dataset, DataLoader  # type: ignore[import-untyped]

import json
from pathlib import Path
from datetime import datetime

from src.models.simple_instruction_processor import SimpleInstructionProcessor


# ============================================================================
# SFT DATASET CLASS
# ============================================================================

class PicoBananaSFTDataset(Dataset):
    """
    Dataset for SFT training on Pico-Banana examples
    """

    def __init__(self, manifest_path: Path, max_length: int = 512):
        """
        Args:
            manifest_path: Path to SFT manifest JSONL file
            max_length: Maximum sequence length for inputs
        """
        self.examples = self._load_manifest(manifest_path)
        self.max_length = max_length

        # Create instruction-to-edit-type mapping
        self.instruction_vocabulary = self._build_vocabulary()

    def _load_manifest(self, manifest_path: Path) -> List[Dict]:
        """Load examples from manifest file"""
        examples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def _build_vocabulary(self) -> Dict[str, int]:
        """Build instruction vocabulary for simple tokenization"""
        instructions = set()
        for example in self.examples:
            # Simple word-level tokenization for demonstration
            text = example['instruction'].lower().replace(',', '').replace('.', '')
            words = text.split()
            instructions.update(words)

        return {
            word: i + 1 for i, word in enumerate(sorted(instructions))
        }  # 0 reserved for padding

    def __len__(self):
        return len(self.examples)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization for demonstration"""
        words = text.lower().replace(',', '').replace('.', '').split()
        tokens = [self.instruction_vocabulary.get(word, 0) for word in words]
        # Pad/truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([0] * (self.max_length - len(tokens)))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize instruction
        instruction_tokens = self._tokenize(example['instruction'])

        # Create target (edit type as classification target)
        # In practice, this would be more sophisticated
        edit_type_categories = {
            'brightness_adjustment': 0,
            'color_adjustment': 1,
            'blur_background': 2
        }
        edit_type = edit_type_categories.get(example['edit_type'], 0)
        target = torch.tensor(edit_type, dtype=torch.long)

        # Quality score (for weighting or filtering)
        quality_score = torch.tensor(example['quality_score'], dtype=torch.float)

        return {
            'instruction_tokens': instruction_tokens,
            'edit_type_target': target,
            'quality_score': quality_score
        }

# ============================================================================
# SFT TRAINING LOOP
# ============================================================================

class PicoBananaSFTTrainer:
    """
    Trainer for SFT phase of modern pipeline
    """

    def __init__(self, config: Dict):
        self.config = config

        # Setup device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else
                                   'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create model, optimizer, loss
        self.model = SimpleInstructionProcessor(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes']
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate']
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            instruction_tokens = batch['instruction_tokens'].to(self.device)
            targets = batch['edit_type_target'].to(self.device)
            quality_scores = batch['quality_score'].to(self.device)

            # Forward pass
            logits = self.model(instruction_tokens)
            loss = self.criterion(logits, targets)

            # Apply quality weighting (higher quality examples get more weight)
            quality_weight = quality_scores.unsqueeze(1)  # [batch_size, 1]
            loss = torch.mean(loss * quality_weight)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test data"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                instruction_tokens = batch['instruction_tokens'].to(self.device)
                targets = batch['edit_type_target'].to(self.device)

                logits = self.model(instruction_tokens)
                loss = self.criterion(logits, targets)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, path)
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"📂 Checkpoint loaded: {path}")


# ============================================================================
# CONFIGURATION & MAIN TRAINING FUNCTION
# ============================================================================

def create_config() -> Dict:
    """Create training configuration"""
    return {
        'batch_size': 2,  # Very small for our 6 examples
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_classes': 3,  # Three edit types in our sample data
        'learning_rate': 1e-3,
        'num_epochs': 5,  # Keep short for demonstration
        'vocab_size': 100,  # We still need to determine this
        'save_dir': './outputs/sft_models',
        'data_dir': './pico_banana_dataset'
    }


def main():
    """Main SFT training function"""
    print("🚀 Pico-Banana-400K SFT Training (Phase 4: Start Small)")
    print("=" * 60)

    config = create_config()

    # Create output directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("📦 Loading SFT dataset...")
    dataset_path = Path(config['data_dir']) / 'manifest_sft.jsonl'
    dataset = PicoBananaSFTDataset(dataset_path)

    # Update vocab size based on actual data
    config['vocab_size'] = max(dataset.instruction_vocabulary.values()) + 1

    print(f"   Found {len(dataset)} examples")
    print(f"   Vocabulary size: {config['vocab_size']}")

    # Create train/val split (simple split for small dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"   Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    # Initialize trainer
    trainer = PicoBananaSFTTrainer(config)

    print("\n🎯 Starting training...")

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n📈 Epoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(".4f")

        # Validate
        val_metrics = trainer.evaluate(val_loader)
        print(".4f")

        # Save best model
        if val_metrics['accuracy'] > trainer.best_accuracy:
            trainer.best_accuracy = val_metrics['accuracy']
            checkpoint_path = save_dir / 'best_model.pt'
            trainer.save_checkpoint(checkpoint_path)

        # Save latest checkpoint
        latest_path = save_dir / 'latest_model.pt'
        trainer.save_checkpoint(latest_path)

    print(f"\n🏆 Training completed! Best validation accuracy: {trainer.best_accuracy:.4f}")

    # Save final configuration
    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("📝 Configuration saved to:", config_path)
    print(".4f")

    # Inference test on our data
    print("\n🔮 Testing inference on sample data...")
    trainer.model.eval()

    # Get a sample example
    example = dataset[0]
    instruction_tokens = example['instruction_tokens'].unsqueeze(0).to(trainer.device)

    with torch.no_grad():
        logits = trainer.model(instruction_tokens)
        prediction = torch.argmax(logits, dim=1).item()

    edit_type_names = {0: 'brightness_adjustment', 1: 'color_adjustment', 2: 'blur_background'}
    predicted_type = edit_type_names.get(int(prediction), 'unknown')

    print(f"   Input: '{dataset.examples[0]['instruction']}'")
    print(f"   Predicted edit type: {predicted_type}")
    print(f"   Actual edit type: {dataset.examples[0]['edit_type']}")

    print("\n✅ SFT training demonstration complete!")
    print("   Next: Would add DPO training for quality improvement")
    print("   Then: Multi-turn training for conversational editing")


if __name__ == "__main__":
    main()
