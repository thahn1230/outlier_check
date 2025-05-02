import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import time
from collections import defaultdict
import warnings
from datasets import load_dataset
import gc
import threading
from tqdm import tqdm
import re

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class LLMMagnitudeAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda", precision: str = "fp16", 
                 output_dir: str = "magnitude_visualizations", layer_limit: int = None):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self.activation_stats = defaultdict(list)
        self.hooks = []
        self.output_dir = output_dir
        self.layer_limit = layer_limit  # Limit analysis to first N layers (speeds up analysis)
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "3d_visualizations"), exist_ok=True)
        
        # Progress tracking
        self.total_components = 0
        self.processed_components = 0
        
    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set torch dtype based on precision
        if self.precision == "fp16":
            torch_dtype = torch.float16
            print("Using FP16 precision")
        else:
            torch_dtype = torch.float32
            print("Using FP32 precision")
        
        # Use device_map="auto" to distribute model across available GPUs
        if self.device == "cuda":
            print("Loading model with auto device mapping (using all available GPUs)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",  # Auto-distribute across GPUs
            )
        else:
            print(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device
            )
        
        self.model.eval()
        print("Model loaded successfully")
        print(f"Model is distributed across devices: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'N/A'}")
        
    def register_hooks(self):
        """Register hooks to capture activations from different layers"""
        def hook_fn(name):
            def hook(module, input, output):
                # Store the input and output values
                input_val = input[0].detach().cpu()
                if isinstance(output, tuple):
                    output_val = output[0].detach().cpu()
                else:
                    output_val = output.detach().cpu()
                
                # Store activations
                self.activation_stats[f"{name}_input"].append(input_val)
                self.activation_stats[f"{name}_output"].append(output_val)
            return hook
        
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # For transformer models, we want to hook attention layers and FFNs
        for name, module in self.model.named_modules():
            # Skip non-leaf modules
            if len(list(module.children())) > 0:
                continue
                
            # Apply layer limit if specified
            if self.layer_limit is not None:
                # Extract layer number if present in name
                layer_match = re.search(r'layers\.(\d+)', name)
                if layer_match and int(layer_match.group(1)) >= self.layer_limit:
                    continue
                
            # Focus on attention components and FFNs
            if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                print(f"Registered hook for: {name}")
    
    def load_example_dataset(self, dataset_name="wikitext", sample_size=5):
        """Load example dataset for activation analysis"""
        try:
            # Different options for datasets
            if dataset_name == "wikitext":
                print(f"Loading wikitext dataset (sample size: {sample_size})...")
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                texts = dataset["text"][:sample_size]
                # Filter out empty texts
                texts = [text for text in texts if len(text.strip()) > 50]
                
            elif dataset_name == "code":
                print(f"Loading code dataset (sample size: {sample_size})...")
                dataset = load_dataset("codeparrot/github-code", split="train")
                texts = [sample["code"] for sample in dataset[:sample_size]]
                
            elif dataset_name == "multilingual":
                print(f"Loading multilingual dataset (sample size: {sample_size})...")
                # Custom multilingual text including Korean
                texts = [
                    "자연어 처리 모델의 가중치 분포를 분석하는 중입니다. 이 텍스트는 다양한 언어 패턴을 포함하고 있습니다.",
                    "自然言語処理モデルの重み分布を分析しています。このテキストには様々な言語パターンが含まれています。",
                    "Analyzing the weight distribution of natural language processing models. This text contains various language patterns.",
                    "Estamos analizando la distribución de pesos en modelos de procesamiento de lenguaje natural.",
                    "Мы анализируем распределение весов в моделях обработки естественного языка."
                ][:sample_size]
                
            else:  # Default to custom text
                print("Using default sample text...")
                texts = [
                    """자연어 처리 모델의 가중치 분포를 분석하는 중입니다. 이 텍스트는 다양한 언어 패턴을 포함하고 있어서 
                    모델이 처리하는 방식을 관찰하기 좋습니다. The distribution of weights in language models can show 
                    interesting patterns across different components like attention and feed-forward networks."""
                ]
                
            print(f"Loaded {len(texts)} text samples")
            return texts
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using default sample text instead")
            return [
                """자연어 처리 모델의 가중치 분포를 분석하는 중입니다. 이 텍스트는 다양한 언어 패턴을 포함하고 있어서 
                모델이 처리하는 방식을 관찰하기 좋습니다. The distribution of weights in language models can show 
                interesting patterns across different components like attention and feed-forward networks."""
            ]
    
    def run_inference(self, texts: List[str]):
        """Run inference on multiple texts to collect activations"""
        self.activation_stats.clear()
        
        print(f"Running inference on {len(texts)} text samples...")
        for i, text in enumerate(texts):
            print(f"Processing text sample {i+1}/{len(texts)}")
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Forward pass with gradient calculation disabled
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Force garbage collection to free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Collected activation statistics for {len(self.activation_stats)} components")
    
    def extract_weights(self) -> Dict[str, torch.Tensor]:
        """Extract weights from different components of the model"""
        weights = {}
        for name, param in self.model.named_parameters():
            # Apply layer limit if specified
            if self.layer_limit is not None:
                layer_match = re.search(r'layers\.(\d+)', name)
                if layer_match and int(layer_match.group(1)) >= self.layer_limit:
                    continue
                    
            # Filter to focus on attention components and FFNs
            if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]):
                weights[name] = param.detach().cpu()
                
        return weights
    
    def safe_numpy_conversion(self, tensor: torch.Tensor) -> np.ndarray:
        """Safely convert tensor to numpy array, handling potential NaN/Inf values"""
        # First to CPU if needed
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
            
        # Convert to numpy
        arr = tensor.numpy()
        
        # Replace NaN/Inf with zeros to avoid computation issues
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return arr
        
    def analyze_magnitude_stats(self, tensor: torch.Tensor) -> Dict:
        """Analyze magnitude statistics of a tensor with numerical stability"""
        # Convert to absolute values for magnitude analysis
        abs_tensor = torch.abs(tensor)
        
        # Check for NaN or Inf values
        if torch.isnan(abs_tensor).any() or torch.isinf(abs_tensor).any():
            print(f"Warning: Tensor contains NaN or Inf values. Replacing them with zeros.")
            abs_tensor = torch.nan_to_num(abs_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to numpy safely
        flat_tensor = self.safe_numpy_conversion(abs_tensor.flatten())
        
        # Ensure we have valid values to analyze
        if len(flat_tensor) == 0 or np.all(flat_tensor == 0):
            return {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99.9": 0.0,
                "dynamic_range": 1.0  # Default to 1.0 for degenerate case
            }
        
        # Calculate statistics
        mean = np.mean(flat_tensor)
        std = np.std(flat_tensor)
        median = np.median(flat_tensor)
        min_val = np.min(flat_tensor)
        max_val = np.max(flat_tensor)
        
        # Calculate percentiles safely
        try:
            p90 = np.percentile(flat_tensor, 90)
            p95 = np.percentile(flat_tensor, 95)
            p99 = np.percentile(flat_tensor, 99)
            p999 = np.percentile(flat_tensor, 99.9)
        except Exception as e:
            print(f"Warning: Error calculating percentiles: {e}. Using fallback values.")
            # Fallback to simple calculation
            sorted_vals = np.sort(flat_tensor)
            n = len(sorted_vals)
            p90 = sorted_vals[min(int(n * 0.9), n-1)] if n > 0 else 0
            p95 = sorted_vals[min(int(n * 0.95), n-1)] if n > 0 else 0
            p99 = sorted_vals[min(int(n * 0.99), n-1)] if n > 0 else 0
            p999 = sorted_vals[min(int(n * 0.999), n-1)] if n > 0 else 0
        
        # Calculate dynamic range safely
        if min_val > 1e-10:
            dynamic_range = max_val / min_val
        else:
            # If min is very close to zero, use a threshold-based approach
            non_zero_values = flat_tensor[flat_tensor > 1e-10]
            if len(non_zero_values) > 0:
                min_non_zero = np.min(non_zero_values)
                dynamic_range = max_val / min_non_zero
            else:
                dynamic_range = 1.0  # Default value when all values are near zero
        
        # Cap dynamic range to avoid extreme values
        dynamic_range = min(dynamic_range, 1e6)
        
        return {
            "mean": float(mean),
            "std": float(std),
            "median": float(median),
            "min": float(min_val),
            "max": float(max_val),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "p99.9": float(p999),
            "dynamic_range": float(dynamic_range)
        }
    
    def visualize_magnitude_distribution(self, tensor: torch.Tensor, title: str, filename: str):
        """Visualize the magnitude distribution of values in a tensor"""
        # Convert to absolute values for magnitude analysis
        abs_tensor = torch.abs(tensor)
        
        # Check for NaN or Inf values
        if torch.isnan(abs_tensor).any() or torch.isinf(abs_tensor).any():
            print(f"Warning: Tensor contains NaN or Inf values. Replacing them with zeros.")
            abs_tensor = torch.nan_to_num(abs_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to numpy safely
        flat_tensor = self.safe_numpy_conversion(abs_tensor.flatten())
        
        # Get magnitude statistics
        stats = self.analyze_magnitude_stats(tensor)
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Main distribution plot with linear scale
            plt.subplot(1, 2, 1)
            hist_data = sns.histplot(flat_tensor, kde=True, stat="density", bins=50)
            
            # Fix for UserWarning: safely add legend items only if percentile lines exist
            lines = []
            labels = []
            
            # Add percentile lines if they exist and are meaningful
            if stats["p99"] > 0:
                line_99 = plt.axvline(x=stats["p99"], color='r', linestyle='--')
                lines.append(line_99)
                labels.append(f'99th percentile: {stats["p99"]:.6f}')
                
            if stats["p99.9"] > 0:
                line_999 = plt.axvline(x=stats["p99.9"], color='g', linestyle='--')
                lines.append(line_999)
                labels.append(f'99.9th percentile: {stats["p99.9"]:.6f}')
            
            # Only add legend if we have items to show
            if lines and labels:
                plt.legend(lines, labels)
            
            plt.title(f"Magnitude Distribution (Linear scale)")
            plt.xlabel("Absolute Value")
            plt.ylabel("Density")
            
            # Log scale distribution - only if we have positive values
            plt.subplot(1, 2, 2)
            if np.any(flat_tensor > 0):
                # Filter out zeros for log scale
                positive_values = flat_tensor[flat_tensor > 0]
                if len(positive_values) > 0:
                    sns.histplot(positive_values, kde=True, stat="density", bins=50, log_scale=(True, False))
                    
                    # Add percentile lines if they exist and are meaningful
                    lines = []
                    labels = []
                    
                    if stats["p99"] > 0:
                        line_99 = plt.axvline(x=stats["p99"], color='r', linestyle='--')
                        lines.append(line_99)
                        labels.append(f'99th percentile: {stats["p99"]:.6f}')
                        
                    if stats["p99.9"] > 0:
                        line_999 = plt.axvline(x=stats["p99.9"], color='g', linestyle='--')
                        lines.append(line_999)
                        labels.append(f'99.9th percentile: {stats["p99.9"]:.6f}')
                    
                    # Only add legend if we have items to show
                    if lines and labels:
                        plt.legend(lines, labels)
            else:
                plt.text(0.5, 0.5, "No positive values for log scale", ha='center', va='center')
                
            plt.title(f"Magnitude Distribution (Log scale)")
            plt.xlabel("Absolute Value (log scale)")
            plt.ylabel("Density")
            
            plt.suptitle(f"{title}\nMean: {stats['mean']:.6f}, Max: {stats['max']:.6f}, Dynamic Range: {stats['dynamic_range']:.2f}x")
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        
        except Exception as e:
            print(f"Error creating visualization for {title}: {e}")
            # If visualization fails, try with a simpler approach
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(flat_tensor, bins=30)
                plt.title(f"Simple histogram for {title} (after error in detailed visualization)")
                plt.xlabel("Absolute Value")
                plt.ylabel("Count")
                plt.savefig(os.path.join(self.output_dir, f"simple_{filename}"))
                plt.close()
            except:
                print(f"Even simple visualization failed for {title}")
        
        # Update progress (progress is now handled in analyze_and_visualize_* methods)
    
    def visualize_3d_channel_heatmap(self, tensor: torch.Tensor, name: str):
        """Create a 3D visualization of channel magnitudes with token-level information"""
        try:
            # 가중치 시각화 건너뛰기 (activation만 분석)
            if "weight" in name and "activation" not in name:
                print(f"Skipping weight visualization for {name} (focusing on activations only)")
                return
                
            # Handle different tensor dimensions
            if len(tensor.shape) < 2:
                print(f"Tensor {name} has insufficient dimensions for 3D visualization")
                return
                
            # For activations: typically [batch, seq_len, channels] or [batch, channels, seq_len]
            # We need to identify which dimensions represent channels and tokens
            
            is_activation = "activation" in name
            
            if is_activation:
                # For activations, the shape is typically:
                # - For inputs: [batch, seq_len, hidden_dim]
                # - For outputs: [batch, seq_len, num_channels] or [batch, num_channels, seq_len]
                
                # Try to determine the dimensions
                if len(tensor.shape) >= 3:
                    # Assume format is [batch, channels/seq_len, seq_len/channels]
                    batch_dim = 0
                    
                    # Heuristic for channel dimension: usually larger than sequence length
                    if tensor.shape[1] > tensor.shape[2]:
                        channel_dim = 1
                        token_dim = 2
                    else:
                        channel_dim = 2
                        token_dim = 1
                    
                    # Extract a single batch for clarity in visualization
                    sample_batch = 0
                    
                    # Reshape to [channels, tokens] for visualization
                    if channel_dim == 1:
                        vis_data = torch.abs(tensor[sample_batch]).transpose(0, 1)  # Now [tokens, channels]
                    else:
                        vis_data = torch.abs(tensor[sample_batch])  # Already [tokens, channels]
                    
                    # Convert to numpy
                    vis_data = self.safe_numpy_conversion(vis_data)
                    
                    # For clearer visualization, transpose to [channels, tokens]
                    vis_data = vis_data.T
                    
                elif len(tensor.shape) == 2:
                    # Assume [batch*seq_len, channels] or [channels, batch*seq_len]
                    if "input" in name or "output" in name:
                        # For activations inputs/outputs, larger dimension is usually tokens
                        if tensor.shape[0] > tensor.shape[1]:
                            vis_data = torch.abs(tensor).T  # [channels, tokens]
                        else:
                            vis_data = torch.abs(tensor)    # Already [channels, tokens]
                    else:
                        # Fallback to default
                        vis_data = torch.abs(tensor)
                    
                    # Convert to numpy
                    vis_data = self.safe_numpy_conversion(vis_data)
            else:
                # 가중치 시각화는 건너뛰기
                print(f"Skipping non-activation visualization for {name}")
                return
            
            # 이제 vis_data의 형태는 [channels, tokens]입니다
            original_shape = vis_data.shape
            print(f"Original data shape for {name}: {original_shape} (channels x tokens)")
            
            # 시각화를 위해 채널(x축)과 토큰(y축)으로 전치합니다
            vis_data_transposed = vis_data.T  # [tokens, channels]
            tokens, channels = vis_data_transposed.shape
            
            print(f"Visualizing full data: {name}: {channels} channels x {tokens} tokens")
            
            # 모든 토큰과 모든 채널을 시각화 (샘플링 없음)
            try:
                # 1. 전체 채널-토큰 히트맵 (무조건 전체 데이터 시각화)
                print(f"Creating full heatmap for {name} with {channels} channels and {tokens} tokens...")
                plt.figure(figsize=(20, 12))
                
                plt.imshow(vis_data_transposed, aspect='auto', cmap='viridis', origin='lower')
                plt.colorbar(label='Absolute Value')
                
                plt.xlabel('Channel Index')
                plt.ylabel('Token Position')
                plt.title(f'Full Channel vs Token Heatmap for {name} ({channels}x{tokens}) - NO SAMPLING')
                
                # 주요 채널 인덱스를 눈금으로 표시 (5% 간격)
                if channels > 20:
                    channel_ticks = np.linspace(0, channels-1, 21).astype(int)
                    plt.xticks(channel_ticks, channel_ticks)
                
                plt.tight_layout()
                full_heatmap_path = os.path.join(self.output_dir, "3d_visualizations", f"{name}_complete_full_heatmap.png")
                print(f"Saving full heatmap to {full_heatmap_path}")
                plt.savefig(full_heatmap_path, dpi=300)
                plt.close()
                
                # 2. 모든 이상치 채널 시각화
                print(f"Detecting outlier channels for {name}...")
                # 각 채널의 평균값 계산
                channel_means = np.mean(vis_data, axis=1)  # [channels]
                
                # 이상치 채널 찾기 (평균 + 3*표준편차 이상)
                channel_mean = np.mean(channel_means)
                channel_std = np.std(channel_means)
                outlier_threshold = channel_mean + 3 * channel_std
                
                outlier_indices = np.where(channel_means > outlier_threshold)[0]
                print(f"Found {len(outlier_indices)} outlier channels in {name}")
                
                if len(outlier_indices) > 0:
                    # 모든 이상치 채널 시각화 (제한 없음)
                    # 내림차순 정렬하여 큰 값을 가진 채널부터 시각화
                    sorted_outlier_indices = outlier_indices[np.argsort(channel_means[outlier_indices])[::-1]]
                    
                    print(f"Visualizing all {len(sorted_outlier_indices)} outlier channels...")
                    plt.figure(figsize=(20, 12))
                    outlier_data = vis_data_transposed[:, sorted_outlier_indices]
                    plt.imshow(outlier_data, aspect='auto', cmap='viridis', origin='lower')
                    plt.colorbar(label='Absolute Value')
                    
                    plt.xlabel('Outlier Channel Index (Sorted by Magnitude)')
                    # x축 눈금을 원래 채널 인덱스로 표시
                    if len(sorted_outlier_indices) <= 50:
                        # 50개 이하면 모든 인덱스 표시
                        plt.xticks(np.arange(len(sorted_outlier_indices)), sorted_outlier_indices)
                    else:
                        # 50개 이상이면 일부만 표시
                        tick_indices = np.linspace(0, len(sorted_outlier_indices)-1, 50).astype(int)
                        plt.xticks(tick_indices, sorted_outlier_indices[tick_indices])
                    
                    plt.ylabel('Token Position')
                    plt.title(f'All Outlier Channels from {name} ({len(sorted_outlier_indices)} channels) - NO SAMPLING')
                    
                    plt.tight_layout()
                    outlier_path = os.path.join(self.output_dir, "3d_visualizations", f"{name}_all_outlier_channels.png")
                    print(f"Saving outlier channels visualization to {outlier_path}")
                    plt.savefig(outlier_path, dpi=300)
                    plt.close()
                    
                    # 3. 상위 50개 이상치 채널에 대한 개별 시각화
                    top_n_outliers = min(50, len(sorted_outlier_indices))
                    top_outliers = sorted_outlier_indices[:top_n_outliers]
                    
                    print(f"Creating individual plots for top {top_n_outliers} outlier channels...")
                    # 개별 채널 디렉토리 생성
                    channel_dir = os.path.join(self.output_dir, "3d_visualizations", f"{name}_individual_channels")
                    os.makedirs(channel_dir, exist_ok=True)
                    
                    for idx, channel_idx in enumerate(top_outliers):
                        # 단일 채널의 모든 토큰 값 추출
                        channel_values = vis_data[channel_idx]
                        
                        plt.figure(figsize=(14, 6))
                        plt.plot(np.arange(len(channel_values)), channel_values)
                        plt.xlabel('Token Position')
                        plt.ylabel('Absolute Value')
                        plt.title(f'Channel {channel_idx} Values (Mean: {np.mean(channel_values):.6f}, Max: {np.max(channel_values):.6f})')
                        plt.grid(True, alpha=0.3)
                        
                        # 평균선 추가
                        plt.axhline(y=np.mean(channel_values), color='r', linestyle='--', 
                                  label=f'Mean: {np.mean(channel_values):.6f}')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(channel_dir, f"channel_{channel_idx}.png"), dpi=200)
                        plt.close()
                        
                        # 진행 상황 표시
                        if (idx+1) % 10 == 0:
                            print(f"  - Processed {idx+1}/{top_n_outliers} individual channel plots")
                    
                    print(f"Individual channel plots saved to {channel_dir}")
                    
                    # 4. 이상치 채널들의 분포 시각화 (히스토그램)
                    plt.figure(figsize=(14, 8))
                    plt.hist(channel_means, bins=100, alpha=0.7)
                    plt.axvline(x=outlier_threshold, color='r', linestyle='--', 
                              label=f'Outlier threshold: {outlier_threshold:.6f}')
                    plt.hist(channel_means[outlier_indices], bins=50, color='r', alpha=0.5, 
                            label=f'Outlier channels ({len(outlier_indices)})')
                    
                    plt.xlabel('Channel Mean Absolute Value')
                    plt.ylabel('Count')
                    plt.title(f'Channel Value Distribution for {name} (Total: {channels} channels)')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", f"{name}_channel_distribution.png"),
                              dpi=300)
                    plt.close()
                    
                    # 5. 채널별 3D 표면도 - 일부 이상치 채널과 비이상치 채널 비교
                    print(f"Creating 3D surface plot for outlier analysis...")
                    
                    # 상위 10개 이상치 채널과 10개 일반 채널 선택
                    top_outlier_channels = sorted_outlier_indices[:10]
                    
                    # 일반 채널 선택 (이상치가 아닌 채널 중에서)
                    non_outlier_indices = np.setdiff1d(np.arange(channels), outlier_indices)
                    # 중간 값을 가진 일반 채널 선택
                    middle_non_outliers = non_outlier_indices[len(non_outlier_indices)//2-5:len(non_outlier_indices)//2+5]
                    
                    # 3D 서피스 플롯 (가장 높은 이상치 채널)
                    for i, channel_idx in enumerate(top_outlier_channels[:5]):  # 상위 5개만 처리
                        channel_data = vis_data[channel_idx]  # [tokens]
                        
                        fig = plt.figure(figsize=(12, 10))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # 토큰 인덱스를 X축, 값을 Z축으로 사용
                        X = np.arange(len(channel_data))
                        Y = np.zeros_like(X)  # Y축은 단일 채널이므로 0으로 고정
                        Z = channel_data
                        
                        # 3D 막대 그래프로 표현 
                        dx = 0.5  # 막대 너비
                        dy = 0.8  # 막대 깊이
                        
                        # 값에 따라 색상 지정
                        norm = plt.Normalize(0, np.max(Z))
                        colors = plt.cm.viridis(norm(Z))
                        
                        for xi, zi, c in zip(X, Z, colors):
                            ax.bar3d(xi-dx/2, -dy/2, 0, dx, dy, zi, color=c, zsort='average', alpha=0.8)
                        
                        # 레이블과 제목
                        ax.set_xlabel('Token Position')
                        ax.set_ylabel('')
                        ax.set_zlabel('Absolute Value')
                        ax.set_title(f'Outlier Channel {channel_idx} (Mean: {np.mean(channel_data):.6f})')
                        
                        # 채널 평균값 표시 (x-z 평면)
                        ax.plot([0, len(channel_data)-1], [0, 0], [np.mean(channel_data), np.mean(channel_data)], 
                               'r--', linewidth=2, label=f'Mean: {np.mean(channel_data):.6f}')
                        
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                              f"{name}_3d_outlier_channel_{channel_idx}.png"), dpi=300)
                        plt.close()
                    
                    # 6. 3D 히트맵 (Token vs Channel Absolute Values)
                    # 모든 이상치 채널과 일부 일반 채널을 3D로 시각화
                    print("Creating 3D heatmap for channel-token-value comparison...")
                    
                    # 전체 데이터가 너무 클 수 있으므로, 일정 간격으로 토큰 샘플링
                    # 하지만 모든 채널은 유지
                    token_step = max(1, tokens // 100)  # 최대 100개 토큰 위치
                    sampled_tokens = np.arange(0, tokens, token_step)
                    
                    # 모든 이상치 채널과 일부 일반 채널 선택
                    analysis_channels = np.concatenate([
                        sorted_outlier_indices[:20],  # 상위 20개 이상치 채널
                        middle_non_outliers[:20]      # 20개 일반 채널
                    ])
                    
                    # 데이터 준비 - [tokens, channels] 형태의 데이터
                    sampled_data = vis_data_transposed[sampled_tokens][:, analysis_channels]
                    
                    # 3D 시각화
                    fig = plt.figure(figsize=(16, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 메쉬그리드 생성 - X축은 채널, Y축은 토큰
                    # meshgrid에서는 첫 번째 인자가 X축, 두 번째 인자가 Y축
                    X, Y = np.meshgrid(np.arange(len(analysis_channels)), np.arange(len(sampled_tokens)))
                    
                    # 서피스 플롯
                    surf = ax.plot_surface(X, Y, sampled_data, cmap='viridis', 
                                         edgecolor='none', alpha=0.8)
                    
                    # 컬러바
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Absolute Value')
                    
                    # 레이블 - X축은 채널, Y축은 토큰
                    ax.set_xlabel('Channel Index', fontsize=14)
                    ax.set_ylabel('Token Position', fontsize=14)
                    ax.set_zlabel('Absolute Value', fontsize=14)
                    
                    # 이상치 채널 영역 표시
                    x_ticks = np.arange(0, len(analysis_channels), 5)  # 5개마다 표시
                    x_labels = [str(analysis_channels[i]) + ('*' if i < 20 else '') for i in x_ticks]
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45, ha='right')
                    
                    # Y축(토큰) 눈금 설정
                    y_ticks = np.arange(0, len(sampled_tokens), len(sampled_tokens)//5)
                    y_labels = [str(sampled_tokens[i]) for i in y_ticks]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    
                    # 설명 추가
                    plt.title('3D Surface: Channel (X) vs Token (Y) vs Absolute Value (Z)', fontsize=16)
                    annotation = "* marks outlier channels\nX-axis: Channel indices (first 20 are outliers, last 20 are non-outliers)\n"
                    annotation += "Y-axis: Token positions\nZ-axis: Absolute activation values"
                    plt.figtext(0.5, 0.01, annotation, ha='center', fontsize=10, 
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
                    
                    # 시점 조정 (보기 좋은 각도로)
                    ax.view_init(elev=30, azim=120)
                    
                    plt.tight_layout(rect=[0, 0.05, 1, 1])
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                          f"{name}_3d_channel_token_value_surface.png"), dpi=300)
                    plt.close()
                    
                    # 추가적인 3D 시각화 - 다른 각도에서 시점
                    fig = plt.figure(figsize=(16, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 동일한 데이터로 다른 각도의 3D 시각화
                    surf = ax.plot_surface(X, Y, sampled_data, cmap='viridis', 
                                         edgecolor='none', alpha=0.8)
                    
                    # 레이블
                    ax.set_xlabel('Channel Index (X)', fontsize=14)
                    ax.set_ylabel('Token Position (Y)', fontsize=14)
                    ax.set_zlabel('Absolute Value (Z)', fontsize=14)
                    
                    # 채널 인덱스 표시 (X축)
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45, ha='right')
                    
                    # 시점 조정 - 채널 축을 따라 보기
                    ax.view_init(elev=20, azim=30)
                    
                    plt.title('3D Surface (Side View): Channel vs Token vs Value', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                          f"{name}_3d_channel_token_value_side.png"), dpi=300)
                    plt.close()
                    
                    # 와이어프레임 3D 시각화 (더 명확한 구조 확인)
                    fig = plt.figure(figsize=(16, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 와이어프레임 플롯
                    wire = ax.plot_wireframe(X, Y, sampled_data, color='blue', 
                                            linewidth=0.5, alpha=0.7)
                    
                    ax.set_xlabel('Channel Index (X)', fontsize=14)
                    ax.set_ylabel('Token Position (Y)', fontsize=14)
                    ax.set_zlabel('Absolute Value (Z)', fontsize=14)
                    
                    # 채널별 구분을 위한 평면 추가
                    for i in range(0, len(analysis_channels), 5):
                        if i < 20:  # 이상치 채널
                            color = 'red'
                            alpha = 0.1
                        else:  # 일반 채널
                            color = 'blue'
                            alpha = 0.05
                        
                        xx = np.array([i, i])
                        yy = np.array([0, len(sampled_tokens)-1])
                        zz = np.zeros((2, 2))
                        ax.plot_surface(xx[:, np.newaxis], yy[np.newaxis, :], zz, color=color, alpha=alpha)
                    
                    ax.view_init(elev=35, azim=70)
                    plt.title('3D Wireframe: Channel vs Token vs Value', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                          f"{name}_3d_channel_token_value_wireframe.png"), dpi=300)
                    plt.close()
                    
                    # 7. 채널 내 값 분석 - 각 이상치 채널에서 값들의 일관성 확인
                    print("Analyzing value consistency within outlier channels...")
                    
                    # 상위 20개 이상치 채널에 대해 분석
                    analysis_channels = sorted_outlier_indices[:20]
                    
                    # 각 채널의 값을 정규화하여 일관성 확인
                    channel_stats = []
                    for ch_idx in analysis_channels:
                        channel_values = vis_data[ch_idx]  # [tokens]
                        ch_mean = np.mean(channel_values)
                        ch_std = np.std(channel_values)
                        ch_min = np.min(channel_values)
                        ch_max = np.max(channel_values)
                        ch_median = np.median(channel_values)
                        ch_ratio = ch_max / (ch_mean + 1e-10)  # 최대값/평균값
                        
                        # 90%, 95%, 99% 백분위수 계산
                        ch_p90 = np.percentile(channel_values, 90)
                        ch_p95 = np.percentile(channel_values, 95)
                        ch_p99 = np.percentile(channel_values, 99)
                        
                        # 값이 평균보다 2배 이상 큰 토큰 비율
                        ratio_above_2x_mean = np.sum(channel_values > 2 * ch_mean) / len(channel_values) * 100
                        
                        channel_stats.append({
                            'channel': ch_idx,
                            'mean': ch_mean,
                            'std': ch_std,
                            'min': ch_min,
                            'max': ch_max,
                            'median': ch_median,
                            'max_to_mean': ch_ratio,
                            'p90': ch_p90,
                            'p95': ch_p95,
                            'p99': ch_p99,
                            'pct_above_2x_mean': ratio_above_2x_mean
                        })
                    
                    # 박스플롯으로 채널 내 값 분포 비교 시각화
                    plt.figure(figsize=(15, 10))
                    channel_values_list = [vis_data[ch_idx] for ch_idx in analysis_channels]
                    plt.boxplot(channel_values_list, labels=analysis_channels)
                    
                    plt.xlabel('Channel Index')
                    plt.ylabel('Absolute Value')
                    plt.title('Value Distribution Within Top Outlier Channels')
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                          f"{name}_outlier_channel_boxplot.png"), dpi=300)
                    plt.close()
                    
                    # 채널별 결과 시각화 (일관성 지표)
                    plt.figure(figsize=(15, 10))
                    
                    # 일관성 지표 - max/mean 비율
                    x = [stat['channel'] for stat in channel_stats]
                    y1 = [stat['max_to_mean'] for stat in channel_stats]
                    y2 = [stat['pct_above_2x_mean'] for stat in channel_stats]
                    
                    ax1 = plt.subplot(2, 1, 1)
                    ax1.bar(x, y1, alpha=0.7)
                    ax1.set_ylabel('Max/Mean Ratio')
                    ax1.set_title('Max to Mean Ratio for Outlier Channels')
                    ax1.axhline(y=2.0, color='r', linestyle='--', label='Ratio = 2')
                    ax1.grid(alpha=0.3)
                    ax1.legend()
                    
                    ax2 = plt.subplot(2, 1, 2)
                    ax2.bar(x, y2, alpha=0.7, color='orange')
                    ax2.set_xlabel('Channel Index')
                    ax2.set_ylabel('% of Tokens')
                    ax2.set_title('% of Tokens with Value > 2x Mean')
                    ax2.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                          f"{name}_outlier_consistency_metrics.png"), dpi=300)
                    plt.close()
                    
                    # 채널 내 값의 분포 히스토그램 비교 (Top 9 channels)
                    plt.figure(figsize=(15, 12))
                    
                    num_plots = min(9, len(analysis_channels))
                    for i in range(num_plots):
                        plt.subplot(3, 3, i+1)
                        ch_idx = analysis_channels[i]
                        channel_values = vis_data[ch_idx]
                        
                        plt.hist(channel_values, bins=50, alpha=0.7)
                        plt.axvline(x=np.mean(channel_values), color='r', linestyle='--', 
                                  label=f'Mean: {np.mean(channel_values):.4f}')
                         
                        plt.title(f'Channel {ch_idx}')
                        if i == 0:
                            plt.legend()
                     
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "3d_visualizations", 
                                          f"{name}_outlier_histogram_comparison.png"), dpi=300)
                    plt.close()
                    
                    # 저장 경로와 요약 정보
                    results_path = os.path.join(self.output_dir, "3d_visualizations", f"{name}_channel_consistency_metrics.txt")
                    
                    # 채널별 일관성 지표를 표로 정리
                    with open(results_path, 'w') as f:
                        f.write(f"Channel Consistency Analysis for {name}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("이 분석은 채널별 outlier가 모든 토큰 위치에서 일관되게 높은 값을 가지는지 확인합니다.\n")
                        f.write("Max/Mean Ratio: 값이 작을수록 채널 내 값들이 일관적인 것을 의미\n")
                        f.write("% > 2x Mean: 평균의 2배 이상인 값의 비율. 낮을수록 이상치가 적고 값이 일관됨\n\n")
                        
                        f.write(f"{'Channel':<10} {'Mean':>10} {'Max':>10} {'Max/Mean':>10} {'% > 2x Mean':>12} {'Std/Mean':>10}\n")
                        f.write("-" * 80 + "\n")
                        
                        consistent_channels = 0
                        for stat in channel_stats:
                            consistent = stat['max_to_mean'] < 3.0 and stat['pct_above_2x_mean'] < 10.0
                            consistent_marker = " *" if consistent else ""
                            consistent_channels += 1 if consistent else 0
                            
                            f.write(f"{stat['channel']:<10} {stat['mean']:>10.6f} {stat['max']:>10.6f} " +
                                   f"{stat['max_to_mean']:>10.2f} {stat['pct_above_2x_mean']:>12.2f}% " +
                                   f"{stat['std']/stat['mean']:>10.2f}{consistent_marker}\n")
                        
                        f.write("\n* 표시는 일관성 있는 outlier 채널을 의미 (max/mean < 3.0 and % > 2x mean < 10%)\n")
                        f.write(f"분석된 {len(channel_stats)} 이상치 채널 중 {consistent_channels}개({consistent_channels/len(channel_stats)*100:.1f}%)가 일관성 있는 outlier입니다.\n")
                    
                    print(f"Channel consistency analysis saved to {results_path}")
                else:
                    print(f"No outlier channels found in {name}")
                
                # 5. 상위 100개 채널 시각화 (이상치가 아니더라도)
                print(f"Visualizing top 100 channels by mean value...")
                top_100_indices = np.argsort(channel_means)[-100:][::-1]  # 평균값 기준 상위 100개
                
                plt.figure(figsize=(20, 12))
                top_100_data = vis_data_transposed[:, top_100_indices]
                plt.imshow(top_100_data, aspect='auto', cmap='viridis', origin='lower')
                plt.colorbar(label='Absolute Value')
                
                plt.xlabel('Top Channel Index (Sorted by Mean Value)')
                # x축 눈금 표시
                if len(top_100_indices) <= 50:
                    plt.xticks(np.arange(len(top_100_indices)), top_100_indices)
                else:
                    tick_indices = np.linspace(0, len(top_100_indices)-1, 50).astype(int)
                    plt.xticks(tick_indices, top_100_indices[tick_indices])
                
                plt.ylabel('Token Position')
                plt.title(f'Top 100 Channels by Mean Value for {name} - NO SAMPLING')
                
                plt.tight_layout()
                top_100_path = os.path.join(self.output_dir, "3d_visualizations", f"{name}_top_100_channels.png")
                print(f"Saving top 100 channels visualization to {top_100_path}")
                plt.savefig(top_100_path, dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error creating visualization for {name}: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in visualization for {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_channel_magnitudes(self, tensor: torch.Tensor, name: str):
        """Analyze if entire channels have consistently large magnitudes (channel-wise outliers)"""
        # Determine which dimension is the channel dimension
        # For weight matrices: [out_channels, in_channels] or [out_channels, in_channels, ...]
        # Channel dimension is typically 0 for weights, 1 for activations
        
        channel_dim = 0 if "weight" in name else 1
        
        # Handle case where tensor doesn't have enough dimensions
        if len(tensor.shape) <= channel_dim:
            print(f"Tensor {name} doesn't have a channel dimension {channel_dim}, skipping channel analysis")
            return None
            
        # Get the number of channels
        num_channels = tensor.shape[channel_dim]
        
        # Compute average magnitude per channel
        channel_magnitudes = []
        channel_max_values = []
        
        try:
            # Compute per-channel statistics based on shape
            if channel_dim == 0:  # For weights (out_channels, in_channels, ...)
                for i in range(num_channels):
                    if len(tensor.shape) == 2:  # Linear layer weights
                        channel_data = tensor[i, :]
                    elif len(tensor.shape) > 2:  # Conv weights or others
                        channel_data = tensor[i, ...]
                    else:
                        continue
                        
                    channel_abs = torch.abs(channel_data)
                    # Replace NaN/Inf with zeros
                    channel_abs = torch.nan_to_num(channel_abs, nan=0.0, posinf=0.0, neginf=0.0)
                    channel_magnitudes.append(torch.mean(channel_abs).item())
                    channel_max_values.append(torch.max(channel_abs).item())
                    
            else:  # For activations or other tensors with channel_dim = 1
                for i in range(num_channels):
                    if len(tensor.shape) == 2:  # 2D tensor [batch, channels]
                        channel_data = tensor[:, i]
                    elif len(tensor.shape) == 3:  # 3D tensor [batch, channels, seq_len]
                        channel_data = tensor[:, i, :]
                    elif len(tensor.shape) == 4:  # 4D tensor
                        channel_data = tensor[:, i, :, :]
                    else:
                        continue
                        
                    channel_abs = torch.abs(channel_data)
                    # Replace NaN/Inf with zeros
                    channel_abs = torch.nan_to_num(channel_abs, nan=0.0, posinf=0.0, neginf=0.0)
                    channel_magnitudes.append(torch.mean(channel_abs).item())
                    channel_max_values.append(torch.max(channel_abs).item())
            
            if not channel_magnitudes:
                return None
                
            # Convert to numpy for easier analysis
            channel_magnitudes = np.array(channel_magnitudes)
            channel_max_values = np.array(channel_max_values)
            
            # Replace NaN/Inf with zeros
            channel_magnitudes = np.nan_to_num(channel_magnitudes, nan=0.0, posinf=0.0, neginf=0.0)
            channel_max_values = np.nan_to_num(channel_max_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate statistics across channels
            mean_magnitude = np.mean(channel_magnitudes)
            std_magnitude = np.std(channel_magnitudes)
            channel_threshold = mean_magnitude + 3 * std_magnitude  # 3 sigma rule
            
            # Identify channels with abnormally high average magnitudes
            channel_outliers = channel_magnitudes > channel_threshold
            num_outlier_channels = np.sum(channel_outliers)
            
            # Prepare results
            results = {
                "tensor_name": name,
                "num_channels": num_channels,
                "channel_magnitudes": channel_magnitudes,
                "channel_max_values": channel_max_values,
                "mean_magnitude": mean_magnitude,
                "std_magnitude": std_magnitude,
                "channel_threshold": channel_threshold,
                "num_outlier_channels": num_outlier_channels,
                "percent_outlier_channels": (num_outlier_channels / num_channels) * 100,
                "outlier_channel_indices": np.where(channel_outliers)[0].tolist()
            }
            
            # Visualize channel magnitude distribution
            self.visualize_channel_magnitudes(results, name)
            
            # Create 3D visualization
            self.visualize_3d_channel_heatmap(tensor, name)
            
            return results
            
        except Exception as e:
            print(f"Error in channel magnitude analysis for {name}: {e}")
            return None
    
    def visualize_channel_magnitudes(self, channel_results: Dict, name: str):
        """Visualize the distribution of channel magnitudes"""
        if channel_results is None:
            return
            
        try:
            channel_magnitudes = channel_results["channel_magnitudes"]
            mean_magnitude = channel_results["mean_magnitude"]
            channel_threshold = channel_results["channel_threshold"]
            outlier_indices = channel_results["outlier_channel_indices"]
            
            plt.figure(figsize=(12, 10))
            
            # Bar plot showing magnitude per channel
            plt.subplot(2, 1, 1)
            bars = plt.bar(range(len(channel_magnitudes)), channel_magnitudes, alpha=0.7)
            
            # Highlight outlier channels
            for idx in outlier_indices:
                bars[idx].set_color('red')
                
            plt.axhline(y=mean_magnitude, color='g', linestyle='-', label=f'Mean: {mean_magnitude:.6f}')
            plt.axhline(y=channel_threshold, color='r', linestyle='--', 
                       label=f'Threshold: {channel_threshold:.6f}')
            
            plt.xlabel("Channel Index")
            plt.ylabel("Average Absolute Magnitude")
            plt.title(f"Channel Magnitudes for {name}")
            plt.legend()
            
            # Distribution of channel magnitudes with outliers marked
            plt.subplot(2, 1, 2)
            sns.histplot(channel_magnitudes, kde=True, bins=30)
            
            # Mark outlier region
            if outlier_indices:
                min_outlier_val = np.min(channel_magnitudes[outlier_indices])
                plt.axvline(x=channel_threshold, color='r', linestyle='--',
                           label=f'Outlier Threshold: {channel_threshold:.6f}')
                
                # Plot outlier distribution if there are any
                if len(outlier_indices) > 1:
                    sns.histplot(channel_magnitudes[outlier_indices], color='r', bins=10, 
                                alpha=0.5, label=f'Outlier Channels ({len(outlier_indices)})')
            
            plt.xlabel("Average Absolute Magnitude")
            plt.ylabel("Count")
            plt.title(f"Distribution of Channel Magnitudes ({channel_results['percent_outlier_channels']:.2f}% are outliers)")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{name}_channel_magnitudes.png"))
            plt.close()
            
            # Create heatmap for the top outlier channels if there are any
            if outlier_indices and len(outlier_indices) > 0:
                self.visualize_top_outlier_channels(channel_results, name)
        
        except Exception as e:
            print(f"Error visualizing channel magnitudes for {name}: {e}")
    
    def visualize_top_outlier_channels(self, channel_results: Dict, name: str, top_n: int = 10):
        """Visualize the distribution of values in the top N outlier channels"""
        try:
            outlier_indices = channel_results["outlier_channel_indices"]
            channel_magnitudes = channel_results["channel_magnitudes"]
            
            # If there are no outliers or fewer than 3, skip
            if not outlier_indices or len(outlier_indices) < 3:
                return
                
            # Limit to top N outliers
            top_n = min(top_n, len(outlier_indices))
            
            # Sort outlier channels by magnitude
            sorted_indices = [idx for _, idx in sorted(
                zip(channel_magnitudes[outlier_indices], outlier_indices), reverse=True)]
            
            top_indices = sorted_indices[:top_n]
            
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(top_indices)), channel_magnitudes[top_indices], alpha=0.7)
            plt.xticks(range(len(top_indices)), [f"Ch {idx}" for idx in top_indices])
            plt.xlabel("Outlier Channel")
            plt.ylabel("Average Absolute Magnitude")
            plt.title(f"Top {top_n} Outlier Channels for {name}")
            
            for i, idx in enumerate(top_indices):
                plt.text(i, channel_magnitudes[idx], f"{channel_magnitudes[idx]:.4f}", 
                        ha='center', va='bottom', rotation=45)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{name}_top_outlier_channels.png"))
            plt.close()
        
        except Exception as e:
            print(f"Error visualizing top outlier channels for {name}: {e}")
    
    def analyze_and_visualize_weights(self):
        """Analyze and visualize weights from the model"""
        weights = self.extract_weights()
        
        print(f"Analyzing weights from {len(weights)} components")
        
        # Group weights by component type
        component_groups = {
            "query": [name for name in weights if "q_proj" in name],
            "key": [name for name in weights if "k_proj" in name],
            "value": [name for name in weights if "v_proj" in name],
            "output": [name for name in weights if "o_proj" in name],
            "ffn_up": [name for name in weights if "up_proj" in name],
            "ffn_down": [name for name in weights if "down_proj" in name],
            "ffn_gate": [name for name in weights if "gate_proj" in name],
        }
        
        # Store channel-level results
        channel_results = {}
        
        # Analyze each weight matrix
        for name, tensor in weights.items():
            print(f"Analyzing weight: {name}, shape: {tensor.shape}")
            
            # Overall magnitude distribution
            self.visualize_magnitude_distribution(
                tensor, 
                f"Weight Magnitude Distribution - {name}", 
                f"weight_magnitude_{name.replace('.', '_')}.png"
            )
            
            # Channel-level analysis
            channel_stats = self.analyze_channel_magnitudes(tensor, f"weight_{name.replace('.', '_')}")
            if channel_stats:
                channel_results[name] = channel_stats
            
            # Update progress after each weight is analyzed
            self.processed_components += 1
            progress = (self.processed_components / self.total_components) * 100 if self.total_components > 0 else 0
            print(f"Progress: {self.processed_components}/{self.total_components} ({progress:.1f}%)")
        
        # Compare distributions across component types
        for group_name, component_list in component_groups.items():
            if not component_list:
                continue
                
            try:
                plt.figure(figsize=(14, 8))
                
                # Show up to 10 components (instead of 5) with a limit on label length for readability
                max_components = min(10, len(component_list))
                for name in component_list[:max_components]:
                    abs_data = torch.abs(weights[name]).flatten()
                    # Handle NaN/Inf
                    abs_data = torch.nan_to_num(abs_data, nan=0.0, posinf=0.0, neginf=0.0)
                    abs_data = self.safe_numpy_conversion(abs_data)
                    
                    # Format label to be more readable
                    short_label = name.split('.')[-3:]  # Get last 3 parts of the name
                    label = '.'.join(short_label)
                    
                    sns.kdeplot(abs_data, label=label)
                
                plt.title(f"Weight Magnitude Distribution Comparison - {group_name} Components")
                plt.xlabel("Absolute Value")
                plt.ylabel("Density")
                plt.legend(fontsize='small')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.output_dir, f"weight_magnitude_comparison_{group_name}.png"))
                plt.close()
                
                # Also plot in log scale
                plt.figure(figsize=(14, 8))
                
                for name in component_list[:max_components]:
                    abs_data = torch.abs(weights[name]).flatten()
                    # Handle NaN/Inf and zeros for log scale
                    abs_data = torch.nan_to_num(abs_data, nan=0.0, posinf=0.0, neginf=0.0)
                    abs_data = self.safe_numpy_conversion(abs_data)
                    # Filter out zeros for log scale
                    abs_data = abs_data[abs_data > 0]
                    
                    # Format label to be more readable
                    short_label = name.split('.')[-3:]  # Get last 3 parts of the name
                    label = '.'.join(short_label)
                    
                    if len(abs_data) > 0:
                        sns.kdeplot(abs_data, label=label)
                
                plt.xscale('log')
                plt.title(f"Weight Magnitude Distribution (Log Scale) - {group_name} Components")
                plt.xlabel("Absolute Value (log scale)")
                plt.ylabel("Density")
                plt.legend(fontsize='small')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.output_dir, f"weight_magnitude_comparison_log_{group_name}.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating comparison plots for {group_name}: {e}")
        
        # Summarize channel-level analysis
        self.summarize_channel_analysis(channel_results, "weights")
    
    def analyze_and_visualize_activations(self):
        """Analyze and visualize activations captured during inference"""
        if not self.activation_stats:
            print("No activations recorded. Run inference first.")
            return
        
        print(f"Analyzing activations from {len(self.activation_stats)} hooks")
        
        # Group activations by component type
        component_groups = {
            "query_input": [name for name in self.activation_stats if "q_proj" in name and "input" in name],
            "query_output": [name for name in self.activation_stats if "q_proj" in name and "output" in name],
            "key_input": [name for name in self.activation_stats if "k_proj" in name and "input" in name],
            "key_output": [name for name in self.activation_stats if "k_proj" in name and "output" in name],
            "value_input": [name for name in self.activation_stats if "v_proj" in name and "input" in name],
            "value_output": [name for name in self.activation_stats if "v_proj" in name and "output" in name],
            "ffn_input": [name for name in self.activation_stats if any(x in name for x in ["up_proj", "down_proj", "gate_proj"]) and "input" in name],
            "ffn_output": [name for name in self.activation_stats if any(x in name for x in ["up_proj", "down_proj", "gate_proj"]) and "output" in name],
        }
        
        # Store channel-level results
        channel_results = {}
        
        # Analyze each activation
        for name, tensors in self.activation_stats.items():
            # We might have multiple tensors from different forward passes
            for i, tensor in enumerate(tensors):
                print(f"Analyzing activation: {name}, pass {i+1}, shape: {tensor.shape}")
                
                # Overall magnitude distribution
                self.visualize_magnitude_distribution(
                    tensor, 
                    f"Activation Magnitude Distribution - {name} (Pass {i+1})", 
                    f"activation_magnitude_{name.replace('.', '_')}_{i+1}.png"
                )
                
                # Channel-level analysis for appropriate tensors
                if len(tensor.shape) > 1:
                    channel_stats = self.analyze_channel_magnitudes(tensor, f"activation_{name.replace('.', '_')}_{i+1}")
                    if channel_stats:
                        channel_results[f"{name}_{i+1}"] = channel_stats
                
                # Only increment the counter once per activation, not for each pass
                # to avoid double-counting activations
                if i == 0:
                    self.processed_components += 1
                    progress = (self.processed_components / self.total_components) * 100 if self.total_components > 0 else 0
                    print(f"Progress: {self.processed_components}/{self.total_components} ({progress:.1f}%)")
        
        # Compare distributions across component types
        for group_name, component_list in component_groups.items():
            if not component_list:
                continue
                
            try:
                plt.figure(figsize=(14, 8))
                
                # Show up to 10 components (instead of 5)
                max_components = min(10, len(component_list))
                for name in component_list[:max_components]:
                    if self.activation_stats[name]:
                        abs_data = torch.abs(self.activation_stats[name][0]).flatten()
                        # Handle NaN/Inf
                        abs_data = torch.nan_to_num(abs_data, nan=0.0, posinf=0.0, neginf=0.0)
                        abs_data = self.safe_numpy_conversion(abs_data)
                        
                        # Format label to be more readable
                        parts = name.split('.')
                        if len(parts) > 3:
                            short_label = parts[-3:]  # Get last 3 parts of the name
                            label = '.'.join(short_label)
                        else:
                            label = name
                        
                        sns.kdeplot(abs_data, label=label)
                
                plt.title(f"Activation Magnitude Distribution Comparison - {group_name} Components")
                plt.xlabel("Absolute Value")
                plt.ylabel("Density")
                plt.legend(fontsize='small')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.output_dir, f"activation_magnitude_comparison_{group_name}.png"))
                plt.close()
                
                # Also plot in log scale
                plt.figure(figsize=(14, 8))
                
                for name in component_list[:max_components]:
                    if self.activation_stats[name]:
                        abs_data = torch.abs(self.activation_stats[name][0]).flatten()
                        # Handle NaN/Inf and zeros for log scale
                        abs_data = torch.nan_to_num(abs_data, nan=0.0, posinf=0.0, neginf=0.0)
                        abs_data = self.safe_numpy_conversion(abs_data)
                        # Filter out zeros for log scale
                        abs_data = abs_data[abs_data > 0]
                        
                        # Format label to be more readable
                        parts = name.split('.')
                        if len(parts) > 3:
                            short_label = parts[-3:]  # Get last 3 parts of the name
                            label = '.'.join(short_label)
                        else:
                            label = name
                        
                        if len(abs_data) > 0:
                            sns.kdeplot(abs_data, label=label)
                
                plt.xscale('log')
                plt.title(f"Activation Magnitude Distribution (Log Scale) - {group_name} Components")
                plt.xlabel("Absolute Value (log scale)")
                plt.ylabel("Density")
                plt.legend(fontsize='small')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.output_dir, f"activation_magnitude_comparison_log_{group_name}.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating comparison plots for {group_name}: {e}")
        
        # Summarize channel-level analysis
        self.summarize_channel_analysis(channel_results, "activations")
    
    def summarize_channel_analysis(self, channel_results: Dict, data_type: str):
        """Create a summary of channel-level analysis across components"""
        if not channel_results:
            return
            
        try:
            # Prepare summary data
            names = []
            total_channels = []
            outlier_channels = []
            outlier_percentages = []
            
            for name, result in channel_results.items():
                names.append(name)
                total_channels.append(result["num_channels"])
                outlier_channels.append(result["num_outlier_channels"])
                outlier_percentages.append(result["percent_outlier_channels"])
            
            # Create summary figure
            plt.figure(figsize=(15, 8))
            
            # Bar chart of outlier percentages
            plt.subplot(1, 2, 1)
            bars = plt.bar(range(len(names)), outlier_percentages)
            plt.xticks(range(len(names)), [name.split('.')[-2:] for name in names], rotation=90)
            plt.xlabel("Component")
            plt.ylabel("Percentage of Outlier Channels (%)")
            plt.title(f"Outlier Channel Percentage by Component ({data_type})")
            
            # Add value labels
            for i, v in enumerate(outlier_percentages):
                plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', rotation=90, fontsize=8)
            
            # Scatter plot of total vs outlier channels
            plt.subplot(1, 2, 2)
            plt.scatter(total_channels, outlier_channels)
            
            # Add component labels
            for i, name in enumerate(names):
                short_name = name.split('.')[-2:]
                plt.annotate(short_name, (total_channels[i], outlier_channels[i]), 
                            fontsize=8, alpha=0.7)
            
            plt.xlabel("Total Channels")
            plt.ylabel("Number of Outlier Channels")
            plt.title(f"Outlier Channels vs Total Channels ({data_type})")
            
            # Add reference line for 10% outliers
            max_channels = max(total_channels)
            plt.plot([0, max_channels], [0, max_channels/10], 'r--', label='10% Line')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{data_type}_channel_analysis_summary.png"))
            plt.close()
            
            # Create a table with the summary statistics
            with open(os.path.join(self.output_dir, f"{data_type}_channel_analysis_summary.txt"), 'w') as f:
                f.write(f"Channel Analysis Summary for {data_type}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"{'Component':<40} {'Total Channels':<15} {'Outlier Channels':<15} {'Percentage':<10}\n")
                f.write("-" * 80 + "\n")
                
                for i, name in enumerate(names):
                    f.write(f"{name:<40} {total_channels[i]:<15} {outlier_channels[i]:<15} {outlier_percentages[i]:.2f}%\n")
        except Exception as e:
            print(f"Error creating channel analysis summary for {data_type}: {e}")
    
    def run_full_analysis(self, dataset_name="wikitext", sample_size=5, layer_limit=None):
        """Run a complete analysis on activations only (weights are skipped)"""
        start_time = time.time()
        self.layer_limit = layer_limit
        
        # Load the model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Register hooks for activation capture
        self.register_hooks()
        
        # Load example dataset
        texts = self.load_example_dataset(dataset_name, sample_size)
        
        # Run inference to capture activations
        self.run_inference(texts)
        
        # Count activations for progress tracking
        activations = len(self.activation_stats)
        
        # Reset progress counters (activations only)
        self.total_components = activations
        self.processed_components = 0
        
        print(f"Starting analysis of {self.total_components} activations (skipping weights)...")
        
        # Skip weight analysis
        print("Skipping weight analysis (focusing on activations only)...")
        
        # Analyze activations
        print("Analyzing activations...")
        self.analyze_and_visualize_activations()
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Analysis complete. Took {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Visualizations saved to {self.output_dir}/")
        
        # Final progress report as sanity check
        print(f"Final progress: {self.processed_components}/{self.total_components} components processed")


# Example usage
if __name__ == "__main__":
    # Set the model to analyze - back to llama-2-7b-hf as requested
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # Use CUDA with automatic device mapping to utilize all 8 GPUs
    device = "cuda"
    
    # Use FP16 precision as requested
    precision = "fp16"
    
    # Layer limit (set to None to analyze all layers, or to a number to limit analysis to first N layers)
    # This helps speed up the analysis
    layer_limit = None  # Analyze all layers
    
    # Create analyzer and run analysis
    analyzer = LLMMagnitudeAnalyzer(model_name, device, precision, layer_limit=layer_limit)
    
    # Run analysis with wikitext dataset (default = 5 samples)
    analyzer.run_full_analysis(dataset_name="wikitext", sample_size=5)
