# PicoTuri EditJudge API Documentation

This document provides comprehensive API documentation for the PicoTuri EditJudge system across all platforms.

## Table of Contents

- [Python API](#python-api)
- [Web API](#web-api)
- [Android API](#android-api)
- [Runtime Engine API](#runtime-engine-api)
- [Adaptation API](#adaptation-api)

## Python API

### Core Components

#### Text Encoder

```python
from src.features_text.bert import BERTTextEmbedder

# Initialize text encoder
text_encoder = BERTTextEmbedder(
    model_name="bert-base-uncased",
    max_length=512,
    device="auto"
)

# Encode text instruction
embedding = text_encoder.encode("Make the image more vibrant")
print(f"Text embedding shape: {embedding.shape}")  # (768,)

# Batch encoding
texts = ["Brighten the photo", "Add more contrast", "Adjust saturation"]
embeddings = text_encoder.encode_batch(texts)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 768)
```

#### Image Encoder

```python
from src.features_image.clip import CLIPImageEmbedder

# Initialize image encoder
image_encoder = CLIPImageEmbedder(
    model_name="ViT-B-32",
    image_size=224,
    device="auto"
)

# Encode single image
embedding = image_encoder.encode("path/to/image.jpg")
print(f"Image embedding shape: {embedding.shape}")  # (512,)

# Encode from PIL Image
from PIL import Image
image = Image.open("path/to/image.jpg")
embedding = image_encoder.encode_pil(image)

# Batch encoding
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
embeddings = image_encoder.encode_batch(image_paths)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 512)
```

#### Fusion Head

```python
from src.fuse.fusion import FusionHead

# Initialize fusion head
fusion_head = FusionHead(
    input_dim=1280,  # 768 (text) + 512 (image)
    hidden_dims=[512, 256, 128],
    output_dim=1,
    dropout=0.1
)

# Load trained weights
fusion_head.load_state_dict(torch.load("models/fusion_head.pt"))
fusion_head.eval()

# Predict quality score
text_embedding = torch.randn(1, 768)
image_embedding = torch.randn(1, 512)
fused_features = torch.cat([text_embedding, image_embedding], dim=1)

with torch.no_grad():
    score = fusion_head(fused_features)
    print(f"Quality score: {score.item():.3f}")  # 0.0 to 1.0
```

### Training API

#### Model Training

```python
from src.train.fusion_trainer import FusionTrainer

# Initialize trainer
trainer = FusionTrainer(
    text_encoder_name="bert-base-uncased",
    image_encoder_name="ViT-B-32",
    fusion_config={
        "hidden_dims": [512, 256, 128],
        "dropout": 0.1
    },
    training_config={
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10
    }
)

# Prepare dataset
dataset = trainer.prepare_dataset(
    data_path="data/training_data.csv",
    text_column="instruction",
    image_column="image_path",
    score_column="quality_score"
)

# Train model
trainer.train(dataset)

# Save model
trainer.save_model("models/editjudge_fusion.pt")
```

#### Model Export

```python
from src.export.onnx_export import ONNXExporter

# Initialize exporter
exporter = ONNXExporter()

# Export text encoder
text_encoder = BERTTextEmbedder("bert-base-uncased")
exporter.export_text_encoder(
    model=text_encoder.model,
    output_path="models/text_encoder.onnx",
    input_shape=(1, 512)
)

# Export image encoder
image_encoder = CLIPImageEmbedder("ViT-B-32")
exporter.export_image_encoder(
    model=image_encoder.model,
    output_path="models/image_encoder.onnx",
    input_shape=(1, 3, 224, 224)
)

# Export fusion head
fusion_head = FusionHead(input_dim=1280, hidden_dims=[512, 256, 128])
exporter.export_fusion_head(
    model=fusion_head,
    output_path="models/fusion_head.onnx",
    input_shape=(1, 1280)
)
```

## Web API

### Model Store

```typescript
// stores/modelStore.ts
import { create } from 'zustand';
import { ort } from 'onnxruntime-web';

interface ModelStore {
  // State
  textEncoder: ort.InferenceSession | null;
  imageEncoder: ort.InferenceSession | null;
  fusionHead: ort.InferenceSession | null;
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  loadModels: () => Promise<void>;
  unloadModels: () => void;
  runInference: (text: string, image: HTMLImageElement) => Promise<number>;
}

const useModelStore = create<ModelStore>((set, get) => ({
  // Initial state
  textEncoder: null,
  imageEncoder: null,
  fusionHead: null,
  isInitialized: false,
  isLoading: false,
  error: null,
  
  // Load models
  loadModels: async () => {
    set({ isLoading: true, error: null });
    
    try {
      // Load ONNX models
      const [textEncoder, imageEncoder, fusionHead] = await Promise.all([
        ort.InferenceSession.create('/models/text_encoder.onnx'),
        ort.InferenceSession.create('/models/image_encoder.onnx'),
        ort.InferenceSession.create('/models/fusion_head.onnx')
      ]);
      
      set({ 
        textEncoder, 
        imageEncoder, 
        fusionHead, 
        isInitialized: true,
        isLoading: false 
      });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },
  
  // Run inference
  runInference: async (text: string, image: HTMLImageElement) => {
    const { textEncoder, imageEncoder, fusionHead } = get();
    
    if (!textEncoder || !imageEncoder || !fusionHead) {
      throw new Error('Models not loaded');
    }
    
    // Preprocess inputs
    const textInputs = await preprocessText(text);
    const imageInputs = await preprocessImage(image);
    
    // Run encoders
    const [textResults, imageResults] = await Promise.all([
      textEncoder.run(textInputs),
      imageEncoder.run(imageInputs)
    ]);
    
    // Fuse features
    const fusedFeatures = concatenateFeatures(
      textResults.last_hidden_state,
      imageResults.image_features
    );
    
    // Run fusion head
    const fusionResults = await fusionHead.run({
      features: fusedFeatures
    });
    
    return fusionResults.scores[0];
  }
}));

export default useModelStore;
```

### Utility Functions

```typescript
// lib/utils.ts

// Text preprocessing
export async function preprocessText(text: string): Promise<ort.Tensor> {
  // Tokenize text (simplified)
  const tokens = tokenize(text);
  const inputIds = new ort.Tensor('int64', tokens, [1, tokens.length]);
  const attentionMask = new ort.Tensor('int64', new Array(tokens.length).fill(1), [1, tokens.length]);
  
  return { input_ids: inputIds, attention_mask: attentionMask };
}

// Image preprocessing
export async function preprocessImage(image: HTMLImageElement): Promise<ort.Tensor> {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Resize to 224x224
  canvas.width = 224;
  canvas.height = 224;
  ctx.drawImage(image, 0, 0, 224, 224);
  
  // Get image data and normalize
  const imageData = ctx.getImageData(0, 0, 224, 224);
  const pixels = new Float32Array(3 * 224 * 224);
  
  for (let i = 0; i < imageData.data.length; i += 4) {
    const pixelIndex = i / 4;
    pixels[pixelIndex] = imageData.data[i] / 255.0;     // R
    pixels[pixelIndex + 224 * 224] = imageData.data[i + 1] / 255.0; // G
    pixels[pixelIndex + 2 * 224 * 224] = imageData.data[i + 2] / 255.0; // B
  }
  
  return { image: new ort.Tensor('float32', pixels, [1, 3, 224, 224]) };
}

// Feature concatenation
export function concatenateFeatures(
  textFeatures: ort.Tensor,
  imageFeatures: ort.Tensor
): ort.Tensor {
  const textData = textFeatures.data;
  const imageData = imageFeatures.data;
  
  const fused = new Float32Array(textData.length + imageData.length);
  fused.set(textData);
  fused.set(imageData, textData.length);
  
  return new ort.Tensor('float32', fused, [1, textData.length + imageData.length]);
}
```

### React Components

```typescript
// components/ResultsDisplay.tsx
import React from 'react';
import useModelStore from '../stores/modelStore';

interface ResultsDisplayProps {
  score: number;
  confidence: number;
  processingTime: number;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  score,
  confidence,
  processingTime
}) => {
  const getScoreColor = (score: number): string => {
    if (score >= 0.7) return 'text-green-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };
  
  const getScoreLabel = (score: number): string => {
    if (score >= 0.7) return 'Excellent';
    if (score >= 0.4) return 'Good';
    return 'Needs Improvement';
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="text-center">
        <div className={`text-4xl font-bold ${getScoreColor(score)}`}>
          {(score * 100).toFixed(1)}%
        </div>
        <div className={`text-lg font-medium ${getScoreColor(score)}`}>
          {getScoreLabel(score)}
        </div>
        <div className="text-sm text-gray-600 mt-2">
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
        <div className="text-sm text-gray-600">
          Processing time: {processingTime}ms
        </div>
      </div>
    </div>
  );
};
```

## Android API

### ViewModel

```kotlin
// viewmodel/EditJudgeViewModel.kt
class EditJudgeViewModel : ViewModel() {
    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()
    
    private val _results = MutableStateFlow<List<AnalysisResult>>(emptyList())
    val results: StateFlow<List<AnalysisResult>> = _results.asStateFlow()
    
    private val _selectedImage = MutableStateFlow<String?>(null)
    val selectedImage: StateFlow<String?> = _selectedImage.asStateFlow()
    
    fun analyzeEdit(instruction: String) {
        viewModelScope.launch {
            if (_selectedImage.value == null) return
            
            _isAnalyzing.value = true
            
            try {
                val startTime = System.currentTimeMillis()
                
                // Run inference (simplified)
                val score = runInference(instruction, _selectedImage.value!!)
                val processingTime = System.currentTimeMillis() - startTime
                
                // Create result
                val result = AnalysisResult(
                    id = System.currentTimeMillis().toString(),
                    score = score,
                    confidence = 0.85f + (Math.random() * 0.15).toFloat(),
                    instruction = instruction,
                    imageUri = _selectedImage.value,
                    processingTime = processingTime.toInt(),
                    memoryUsage = (50 + Math.random() * 100).toInt(),
                    timestamp = System.currentTimeMillis()
                )
                
                _results.value = listOf(result) + _results.value.take(9)
                
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isAnalyzing.value = false
            }
        }
    }
    
    private suspend fun runInference(instruction: String, imageUri: String): Float {
        // Mock inference - in real implementation, use ONNX Runtime
        delay(170) // Simulate processing time
        return (Math.random() * 0.6 + 0.4).toFloat()
    }
    
    fun setSelectedImage(imageUri: String) {
        _selectedImage.value = imageUri
    }
}
```

### Main Activity

```kotlin
// MainActivity.kt
@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    private val viewModel: EditJudgeViewModel by viewModels()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Load ONNX Runtime native library
        try {
            OrtEnvironment.getEnvironment()
        } catch (e: Exception) {
            Log.e("EditJudge", "Failed to load ONNX Runtime", e)
        }
        
        setContent {
            EditJudgeTheme {
                EditJudgeApp(viewModel = viewModel)
            }
        }
    }
}
```

### UI Components

```kotlin
// screens/UploadScreen.kt
@Composable
fun UploadScreen(
    viewModel: EditJudgeViewModel,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var selectedImageUri by remember { mutableStateOf<String?>(null) }
    var instructionText by remember { mutableStateOf("") }
    val isAnalyzing by viewModel.isAnalyzing.collectAsState()
    
    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { 
            selectedImageUri = it.toString()
            viewModel.setSelectedImage(it.toString())
        }
    }
    
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Image upload section
        ImageUploadCard(
            imageUri = selectedImageUri,
            onPickImage = { imagePickerLauncher.launch("image/*") },
            modifier = Modifier.fillMaxWidth()
        )
        
        // Text input section
        TextInputCard(
            text = instructionText,
            onTextChanged = { instructionText = it },
            modifier = Modifier.fillMaxWidth()
        )
        
        // Analyze button
        Button(
            onClick = {
                if (selectedImageUri != null && instructionText.isNotBlank()) {
                    scope.launch {
                        viewModel.analyzeEdit(instructionText)
                    }
                }
            },
            enabled = selectedImageUri != null && instructionText.isNotBlank() && !isAnalyzing,
            modifier = Modifier.fillMaxWidth()
        ) {
            if (isAnalyzing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Analyzing...")
            } else {
                Icon(Icons.Default.Analytics, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Analyze Edit Quality")
            }
        }
    }
}
```

## Runtime Engine API

### Session Pool

```python
from src.runtime.engine import SessionPool, RuntimeEngine

# Initialize session pool
session_pool = SessionPool(
    model_paths={
        "text_encoder": "models/text_encoder.onnx",
        "image_encoder": "models/image_encoder.onnx",
        "fusion_head": "models/fusion_head.onnx"
    },
    pool_size=4,
    warm_up=True
)

# Initialize session pool
await session_pool.initialize()

# Get session context
async with session_pool.get_session_context("text_encoder") as session_info:
    # Run inference
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    outputs = session_info.session.run(None, inputs)
    
# Get pool statistics
stats = session_pool.get_pool_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Average inference time: {stats['avg_inference_time']:.2f}ms")

# Shutdown session pool
await session_pool.shutdown()
```

### Runtime Engine

```python
# Initialize runtime engine
engine = RuntimeEngine(
    model_paths={
        "text_encoder": "models/text_encoder.onnx",
        "image_encoder": "models/image_encoder.onnx",
        "fusion_head": "models/fusion_head.onnx"
    },
    pool_size=4,
    batch_strategy=BatchStrategy.ADAPTIVE,
    enable_batching=True
)

# Initialize engine
await engine.initialize()

# Run text encoder
text_embedding = await engine.run_text_encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    request_id="text_001"
)

# Run image encoder
image_embedding = await engine.run_image_encoder(
    image=image_tensor,
    request_id="image_001"
)

# Run fusion head
score = await engine.run_fusion_head(
    features=fused_features,
    request_id="fusion_001"
)

# Get engine statistics
stats = engine.get_engine_stats()
print(f"Engine stats: {stats}")

# Shutdown engine
await engine.shutdown()
```

### Adaptive Micro-Batcher

```python
from src.runtime.batcher import AdaptiveMicroBatcher, BatchStrategy

# Create batcher
batcher = AdaptiveMicroBatcher(
    strategy=BatchStrategy.ADAPTIVE,
    min_batch_size=1,
    max_batch_size=32,
    target_latency_ms=50.0,
    max_wait_time_ms=10.0
)

# Define processing function
async def process_batch(batch_data):
    # Process batch of requests
    results = []
    for data in batch_data:
        # Run inference
        result = await run_inference(data)
        results.append(result)
    return results

# Start batcher
await batcher.start(process_batch)

# Submit request
result = await batcher.submit_request(
    request_id="req_001",
    data={"text": "Make image brighter", "image": image_data}
)

# Get metrics
metrics = batcher.get_metrics()
print(f"Batcher metrics: {metrics}")

# Stop batcher
await batcher.stop()
```

## Adaptation API

### LoRA Adapter

```python
from src.adaptation.lora import LoRAAdapter, create_lora_config, LoRARank

# Create LoRA configuration
lora_config = create_lora_config(
    rank=LoRARank.MEDIUM,
    alpha=32.0,
    dropout=0.1
)

# Create LoRA adapter
adapter = LoRAAdapter(
    base_model_name="bert-base-uncased",
    lora_config=lora_config,
    device="auto"
)

# Enable training
adapter.enable_training()

# Fine-tune on domain data
trainer = LoRATrainer(
    adapter=adapter,
    learning_rate=1e-4,
    max_steps=1000
)

# Train adapter
for epoch in range(10):
    for batch in domain_dataloader:
        loss = trainer.train_step(batch, loss_fn)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save adapter
adapter.save_adapter("adapters/my_domain")

# Load adapter
loaded_adapter = LoRAAdapter.load_adapter("adapters/my_domain")
```

### Domain Calibration

```python
from src.adaptation.calibration import DomainAdapter, CalibrationConfig, CalibrationMethod

# Create calibration configuration
calib_config = CalibrationConfig(
    method=CalibrationMethod.TEMPERATURE_SCALING,
    temperature_init=1.0
)

drift_config = DriftDetectionConfig(
    method=DriftDetectionMethod.KS_TEST,
    significance_level=0.05
)

# Create domain adapter
adapter = DomainAdapter(calib_config, drift_config)

# Calibrate on validation data
scores = model.predict_proba(validation_data)
targets = validation_data["quality_score"]

metrics = adapter.calibrate(scores, targets)
print(f"ECE: {metrics.ece:.4f}")
print(f"Brier score: {metrics.brier_score:.4f}")

# Predict calibrated probabilities
test_scores = model.predict_proba(test_data)
calibrated_probs = adapter.predict_proba(test_scores)

# Detect drift
drift_metrics = adapter.detect_drift(new_scores)
if drift_metrics.drift_detected:
    print(f"Drift detected! p-value: {drift_metrics.p_value:.4f}")

# Save adapter
adapter.save("calibrators/my_domain")
```

### Parameter Estimation

```python
from src.adaptation.lora import estimate_lora_parameters

# Estimate LoRA parameters
base_model = AutoModel.from_pretrained("bert-base-uncased")
lora_config = create_lora_config(rank=16)

param_info = estimate_lora_parameters(base_model, lora_config)
print(f"Base parameters: {param_info['base_parameters']:,}")
print(f"LoRA parameters: {param_info['lora_parameters']:,}")
print(f"LoRA percentage: {param_info['lora_percentage']:.2f}%")
```

## Error Handling

### Common Errors

```python
# Model loading errors
try:
    model = BERTTextEmbedder("bert-base-uncased")
except OSError as e:
    print(f"Model not found: {e}")
except RuntimeError as e:
    print(f"Model initialization failed: {e}")

# Inference errors
try:
    embedding = text_encoder.encode(text)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Inference failed: {e}")

# Runtime engine errors
try:
    await engine.initialize()
except Exception as e:
    print(f"Engine initialization failed: {e}")
```

### Performance Monitoring

```python
# Enable performance monitoring
engine = RuntimeEngine(
    model_paths=model_paths,
    enable_monitoring=True
)

# Get performance metrics
stats = engine.get_engine_stats()
print(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Throughput: {stats['avg_throughput']:.1f} samples/s")
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
```

## Configuration

### Environment Configuration

```python
# Load configuration from YAML
import yaml

with open("configs/project.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use configuration
model_config = config["models"]
training_config = config["training"]
runtime_config = config["runtime"]

# Initialize components with config
text_encoder = BERTTextEmbedder(
    model_name=model_config["text_encoder"]["model_name"],
    max_length=model_config["text_encoder"]["max_length"]
)
```

### Runtime Configuration

```python
# Configure batcher
batcher = AdaptiveMicroBatcher(
    strategy=BatchStrategy(runtime_config["batcher"]["strategy"]),
    min_batch_size=runtime_config["batcher"]["min_batch_size"],
    max_batch_size=runtime_config["batcher"]["max_batch_size"],
    target_latency_ms=runtime_config["batcher"]["target_latency_ms"]
)

# Configure session pool
session_pool = SessionPool(
    model_paths=model_paths,
    pool_size=runtime_config["session_pool"]["pool_size"],
    warm_up=runtime_config["session_pool"]["warm_up"]
)
```

This API documentation provides comprehensive coverage of all PicoTuri EditJudge components across different platforms. For more detailed examples and advanced usage, please refer to the example implementations in the repository.
