import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import warnings
from datasets import load_dataset
import random

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def load_wikitext_samples(num_samples=5, min_length=100, max_length=500):
    """WikiText 데이터셋에서 샘플 텍스트를 로드합니다"""
    print(f"Loading {num_samples} samples from WikiText dataset...")
    
    # WikiText-2 데이터셋 로드
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # 비어있지 않은 텍스트만 필터링
    non_empty_texts = [text for text in dataset["text"] if len(text.strip()) >= min_length]
    
    # 랜덤하게 샘플링
    sampled_texts = []
    for _ in range(num_samples):
        text = random.choice(non_empty_texts)
        # 텍스트가 너무 길면 잘라내기
        if len(text) > max_length:
            text = text[:max_length]
        sampled_texts.append(text)
    
    print(f"Loaded {len(sampled_texts)} text samples from WikiText")
    return sampled_texts

class ImprovedOutlierAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda", precision="fp16", 
                 output_dir="magnitude_visualizations"):
        """초기화 함수"""
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self.activation_stats = {}
        self.hooks = []
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_output_dir(self, name: str, text_index: int) -> str:
        """레이어 및 모듈별 결과 저장 위치 경로 생성"""
        # 텍스트 인덱스별 디렉토리 생성
        text_dir = os.path.join(self.output_dir, f"text_{text_index+1}")
        os.makedirs(text_dir, exist_ok=True)
        
        layer_match = re.search(r'layers\.(\d+)', name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            layer_dir = os.path.join(text_dir, f"layer_{layer_num}")
            
            # 모듈 타입 추출
            module_type = None
            if "self_attn" in name:
                if "q_proj" in name:
                    module_type = "attention_q_proj"
                elif "k_proj" in name:
                    module_type = "attention_k_proj"
                elif "v_proj" in name:
                    module_type = "attention_v_proj"
                elif "o_proj" in name:
                    module_type = "attention_o_proj"
                else:
                    module_type = "attention_other"
            elif "mlp" in name or "up_proj" in name or "down_proj" in name or "gate_proj" in name:
                if "up_proj" in name:
                    module_type = "mlp_up_proj"
                elif "down_proj" in name:
                    module_type = "mlp_down_proj"
                elif "gate_proj" in name:
                    module_type = "mlp_gate_proj"
                else:
                    module_type = "mlp_other"
            elif "input_layernorm" in name:
                module_type = "input_layernorm"
            elif "post_attention_layernorm" in name:
                module_type = "post_attention_layernorm"
            else:
                module_type = "other"
                
            if module_type:
                module_dir = os.path.join(layer_dir, module_type)
                os.makedirs(module_dir, exist_ok=True)
                return module_dir
            else:
                os.makedirs(layer_dir, exist_ok=True)
                return layer_dir
        else:
            # 레이어와 연관되지 않은 컴포넌트
            other_dir = os.path.join(text_dir, "other_components")
            os.makedirs(other_dir, exist_ok=True)
            return other_dir
            
    def load_model(self):
        """모델 및 토크나이저 로드"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 정밀도 설정
        if self.precision == "fp16":
            torch_dtype = torch.float16
            print("Using FP16 precision")
        else:
            torch_dtype = torch.float32
            print("Using FP32 precision")
        
        # 자동 장치 매핑으로 모든 가용 GPU 활용
        if self.device == "cuda":
            print("Loading model with auto device mapping (using all available GPUs)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
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
        """각 레이어와 모듈에 후크 등록"""
        def hook_fn(name):
            def hook(module, input, output):
                # 각 모듈의 입력과 출력 텐서 캡처
                input_val = input[0].detach().cpu()
                if isinstance(output, tuple):
                    output_val = output[0].detach().cpu()
                else:
                    output_val = output.detach().cpu()
                
                # 입력과 출력 저장
                self.activation_stats[f"{name}_input"] = input_val
                self.activation_stats[f"{name}_output"] = output_val
            return hook

        # 기존 후크 제거
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # 모든 모듈에 후크 등록
        for name, module in self.model.named_modules():
            # 자식이 있는 컴포지트 모듈 건너뛰기
            if len(list(module.children())) > 0:
                continue

            # 주요 모듈에 후크 등록
            if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "input_layernorm", "post_attention_layernorm"]):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                print(f"Registered hook for: {name}")
    
    def run_inference(self, text):
        """입력 텍스트에 대해 모델 추론 실행하고 활성화 데이터 수집"""
        print(f"Running inference on input text...")
        
        # 데이터 사전 클리어
        self.activation_stats.clear()
        
        # 토큰화 및 모델 실행
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        print(f"Input length: {len(inputs['input_ids'][0])} tokens")
        
        # 그래디언트 계산 없이 순전파
        with torch.no_grad():
            outputs = self.model(**inputs.to(next(self.model.parameters()).device))
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 결과 확인
        num_components = len(self.activation_stats)
        print(f"Collected activation data for {num_components} components")
    
    def safe_numpy_conversion(self, tensor):
        """텐서를 안전하게 numpy 배열로 변환"""
        # CPU로 이동
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
            
        # numpy로 변환
        arr = tensor.numpy()
        
        # NaN/Inf 값 처리
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return arr
    
    def visualize_flat_channel_heatmap(self, name, text_index):
        """채널을 일자로 그려주는 히트맵 시각화 생성 (x=channel index, y=token index, z=absolute value)"""
        if name not in self.activation_stats:
            print(f"No data found for {name}")
            return
            
        try:
            # 텐서 가져오기
            tensor = self.activation_stats[name]
            
            # 출력 디렉토리 생성
            output_dir = self.get_output_dir(name, text_index)
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 각 outlier 채널별 개별 분석을 위한 디렉토리
            outlier_detail_dir = os.path.join(viz_dir, "outlier_details")
            os.makedirs(outlier_detail_dir, exist_ok=True)
            
            # 텐서 형태 확인
            print(f"Tensor shape for {name}: {tensor.shape}")
            
            # 텐서 모양 처리 - [batch, seq_len, channels] 또는 [batch, channels, seq_len] 가정
            if len(tensor.shape) >= 3:
                # [batch, seq_len, hidden_dim] 형태 (LLaMA 기본 구조)
                if tensor.shape[1] <= tensor.shape[2]:
                    tokens = tensor.shape[1]
                    channels = tensor.shape[2]
                    # [tokens, channels] 형태로 변환 (채널이 x축, 토큰이 y축)
                    vis_data = torch.abs(tensor[0])
                else:
                    # [batch, hidden_dim, seq_len] 형태
                    channels = tensor.shape[1]
                    tokens = tensor.shape[2]
                    # [tokens, channels] 형태로 변환 (채널이 x축, 토큰이 y축)
                    vis_data = torch.abs(tensor[0].transpose(0, 1))
            elif len(tensor.shape) == 2:
                # 2D 텐서는 [seq_len, hidden_dim] 또는 [hidden_dim, seq_len] 형태로 가정
                if tensor.shape[0] <= tensor.shape[1]:
                    tokens = tensor.shape[0]
                    channels = tensor.shape[1]
                    # [tokens, channels] 형태 (채널이 x축, 토큰이 y축)
                    vis_data = torch.abs(tensor)
                else:
                    channels = tensor.shape[0]
                    tokens = tensor.shape[1]
                    # [tokens, channels] 형태로 변환 (채널이 x축, 토큰이 y축)
                    vis_data = torch.abs(tensor.transpose(0, 1))
            else:
                print(f"Unsupported tensor shape for {name}: {tensor.shape}")
                return
                
            # 안전하게 numpy로 변환
            vis_data_np = self.safe_numpy_conversion(vis_data)
            
            print(f"Visualization shape: {vis_data_np.shape} (tokens x channels)")
            
            # 1. 기본 히트맵 - X축: 채널 인덱스, Y축: 토큰 인덱스
            plt.figure(figsize=(20, 10))
            plt.imshow(vis_data_np, aspect='auto', cmap='viridis')
            plt.colorbar(label='Absolute Value')
            plt.xlabel('Channel Index')
            plt.ylabel('Token Index')
            plt.title(f'Channel vs Token Heatmap for {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{name.replace('.', '_')}_heatmap.png"), dpi=300)
            plt.close()
            
            # 채널이 너무 많아 한 번에 시각화하기 어려운 경우, 그룹으로 분할하여 시각화
            # 각 그룹은 1000개 채널씩 포함
            group_size = 1000
            num_groups = (channels + group_size - 1) // group_size
            
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = min((g + 1) * group_size, channels)
                
                # 각 그룹별 데이터 추출
                group_data = vis_data_np[:, start_idx:end_idx]
                
                # 그룹별 히트맵
                plt.figure(figsize=(20, 10))
                plt.imshow(group_data, aspect='auto', cmap='viridis')
                plt.colorbar(label='Absolute Value')
                plt.xlabel(f'Channel Index (Group {g+1}: {start_idx}-{end_idx-1})')
                plt.ylabel('Token Index')
                plt.title(f'Channel Group {g+1} Heatmap for {name}')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{name.replace('.', '_')}_group_{g+1}_heatmap.png"), dpi=300)
                plt.close()
            
            # 2. 채널 평균값 및 이상치 계산
            channel_means = np.mean(vis_data_np, axis=0)  # 각 채널의 평균값
            channel_mean = np.mean(channel_means)  # 전체 평균
            channel_std = np.std(channel_means)  # 표준편차
            outlier_threshold = channel_mean + 3 * channel_std  # 이상치 임계값
            
            # 이상치 채널 식별
            outlier_indices = np.where(channel_means > outlier_threshold)[0]
            print(f"Found {len(outlier_indices)} outlier channels in {channels} total channels")
            
            # 채널 평균 그래프
            plt.figure(figsize=(20, 8))
            plt.plot(np.arange(channels), channel_means, 'b-', alpha=0.5)
            plt.axhline(y=outlier_threshold, color='r', linestyle='--', label=f'Outlier Threshold: {outlier_threshold:.4f}')
            plt.xlabel('Channel Index')
            plt.ylabel('Mean Absolute Value')
            plt.title(f'Channel Mean Values for {name}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{name.replace('.', '_')}_channel_means.png"), dpi=300)
            plt.close()
            
            # 이상치 채널 목록 저장
            with open(os.path.join(viz_dir, f"{name.replace('.', '_')}_outliers.txt"), 'w') as f:
                f.write(f"Outlier channels for {name}:\n")
                f.write(f"Total outliers: {len(outlier_indices)} of {channels} channels\n")
                f.write(f"Outlier threshold: mean + 3*std = {channel_mean:.6f} + 3*{channel_std:.6f} = {outlier_threshold:.6f}\n\n")
                f.write("Channel Index: Mean Value\n")
                for idx in outlier_indices:
                    f.write(f"{idx}: {channel_means[idx]:.6f}\n")
            
            # 3. 상위 이상치 채널에 대한 토큰별 값 라인 플롯
            if len(outlier_indices) > 0:
                # 상위 10개 이상치만 표시
                top_outliers = outlier_indices[np.argsort(-channel_means[outlier_indices])][:10]
                
                plt.figure(figsize=(15, 10))
                for idx in top_outliers:
                    plt.plot(np.arange(tokens), vis_data_np[:, idx], 
                           label=f'Channel {idx} (Mean: {channel_means[idx]:.4f})')
                
                plt.xlabel('Token Index')
                plt.ylabel('Absolute Value')
                plt.title(f'Top 10 Outlier Channels Across Tokens for {name}')
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{name.replace('.', '_')}_top_outliers.png"))
                plt.close()
                
                # 4. 이상치 채널만 히트맵으로 표시
                if len(outlier_indices) > 1:
                    # 이상치 채널 데이터 추출
                    outlier_data = vis_data_np[:, outlier_indices]
                    
                    plt.figure(figsize=(20, 10))
                    plt.imshow(outlier_data, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Absolute Value')
                    plt.xlabel('Outlier Channel Index')
                    plt.ylabel('Token Index')
                    plt.title(f'Outlier Channels Heatmap for {name}')
                    
                    # x축에 실제 채널 인덱스 표시
                    num_ticks = min(20, len(outlier_indices))
                    tick_indices = np.linspace(0, len(outlier_indices)-1, num_ticks, dtype=int)
                    plt.xticks(tick_indices, [outlier_indices[i] for i in tick_indices])
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f"{name.replace('.', '_')}_outlier_heatmap.png"), dpi=300)
                    plt.close()
                
                # 5. 새로 추가: 각 outlier 채널별 개별 시각화 - 토큰 인덱스에 따른 값 변화
                print(f"Generating individual visualizations for {len(outlier_indices)} outlier channels...")
                
                # 상위 50개까지만 시각화 (너무 많은 경우 제한)
                max_outliers_to_visualize = min(50, len(outlier_indices))
                top_outliers_detailed = outlier_indices[np.argsort(-channel_means[outlier_indices])][:max_outliers_to_visualize]
                
                # 토큰 ID를 가져오기 (시각화에 활용)
                token_ids = None
                token_labels = None
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    try:
                        # 마지막으로 사용된 입력 ID를 가져오기 시도
                        # 이 부분은 모델마다 달라질 수 있으므로 일반적인 방식으로 구현
                        token_labels = list(range(tokens))
                    except Exception as e:
                        print(f"Could not retrieve token labels: {e}")
                        token_labels = list(range(tokens))
                else:
                    token_labels = list(range(tokens))
                
                # 각 outlier 채널별 개별 시각화
                for i, idx in enumerate(top_outliers_detailed):
                    try:
                        # 채널값 추출
                        channel_values = vis_data_np[:, idx]
                        
                        # 1. 라인 그래프 - 토큰 인덱스별 값 변화
                        plt.figure(figsize=(15, 6))
                        plt.plot(np.arange(tokens), channel_values, 'b-', marker='o', markersize=4)
                        plt.xlabel('Token Index')
                        plt.ylabel('Activation Value (Absolute)')
                        plt.title(f'Channel {idx} Values Across Tokens (Mean: {channel_means[idx]:.4f})')
                        plt.grid(alpha=0.3)
                        
                        # 평균값 및 표준편차 표시
                        mean_val = np.mean(channel_values)
                        std_val = np.std(channel_values)
                        plt.axhline(y=mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')
                        plt.axhline(y=mean_val + 2*std_val, color='g', linestyle=':', label=f'Mean + 2*STD')
                        plt.axhline(y=mean_val - 2*std_val, color='g', linestyle=':', label=f'Mean - 2*STD')
                        
                        # 상위 N개 값을 강조 표시
                        top_n = 5
                        top_indices = np.argsort(-channel_values)[:top_n]
                        plt.scatter(top_indices, channel_values[top_indices], color='red', s=100, 
                                   label=f'Top {top_n} Values', zorder=5)
                        
                        for j, idx_top in enumerate(top_indices):
                            plt.annotate(f"Token {idx_top}", 
                                        (idx_top, channel_values[idx_top]),
                                        xytext=(5, 5), textcoords='offset points')
                        
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(outlier_detail_dir, f"{name.replace('.', '_')}_channel_{idx}_line.png"), dpi=300)
                        plt.close()
                        
                        # 2. 바 그래프 - 상위 토큰값만 시각화
                        top_n_bar = min(20, tokens)  # 상위 20개 토큰만 표시
                        top_indices_bar = np.argsort(-channel_values)[:top_n_bar]
                        
                        plt.figure(figsize=(15, 8))
                        colors = plt.cm.viridis(np.linspace(0, 1, top_n_bar))
                        bars = plt.bar(range(top_n_bar), channel_values[top_indices_bar], color=colors)
                        plt.xlabel('Top Tokens (by activation value)')
                        plt.ylabel('Activation Value (Absolute)')
                        plt.title(f'Top {top_n_bar} Tokens for Channel {idx}')
                        
                        # x축 레이블을 토큰 인덱스로 표시
                        plt.xticks(range(top_n_bar), [f"{top_indices_bar[i]}" for i in range(top_n_bar)], rotation=45)
                        
                        # 바 위에 값 표시
                        for bar_idx, bar in enumerate(bars):
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(channel_values),
                                   f'{channel_values[top_indices_bar[bar_idx]]:.3f}',
                                   ha='center', va='bottom', rotation=45, fontsize=8)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(outlier_detail_dir, f"{name.replace('.', '_')}_channel_{idx}_top_tokens.png"), dpi=300)
                        plt.close()
                        
                        # 3. 히트맵 - 1D 데이터를 2D 매트릭스로 재구성
                        # 이 시각화는 토큰들을 그리드 형태로 배치하여 패턴을 보기 쉽게 함
                        grid_size = int(np.ceil(np.sqrt(tokens)))
                        grid_data = np.zeros((grid_size, grid_size))
                        
                        for t in range(tokens):
                            row = t // grid_size
                            col = t % grid_size
                            if row < grid_size and col < grid_size:
                                grid_data[row, col] = channel_values[t]
                        
                        plt.figure(figsize=(10, 8))
                        plt.imshow(grid_data, cmap='viridis')
                        plt.colorbar(label='Activation Value')
                        plt.xlabel('Token Grid Column')
                        plt.ylabel('Token Grid Row')
                        plt.title(f'Channel {idx} Token Values as 2D Grid')
                        plt.tight_layout()
                        plt.savefig(os.path.join(outlier_detail_dir, f"{name.replace('.', '_')}_channel_{idx}_grid.png"), dpi=300)
                        plt.close()
                        
                        # 4. CSV 파일로 데이터 내보내기 (추후 분석용)
                        csv_dir = os.path.join(outlier_detail_dir, "csv_data")
                        os.makedirs(csv_dir, exist_ok=True)
                        
                        # 채널 데이터와 토큰 인덱스를 CSV로 저장
                        channel_data_dict = {
                            'token_index': list(range(tokens)),
                            'activation_value': channel_values.tolist()
                        }
                        
                        # numpy를 사용하여 CSV 저장
                        csv_data = np.column_stack((np.arange(tokens), channel_values))
                        np.savetxt(
                            os.path.join(csv_dir, f"{name.replace('.', '_')}_channel_{idx}_data.csv"), 
                            csv_data, 
                            delimiter=',', 
                            header='token_index,activation_value',
                            comments=''
                        )
                        
                    except Exception as e:
                        print(f"Error visualizing channel {idx}: {e}")
            
        except Exception as e:
            print(f"Error in visualization for {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def run_analysis(self, input_texts):
        """주어진 텍스트들에 대한 전체 분석 실행"""
        start_time = time.time()
        
        # 1. 모델 로드
        if self.model is None:
            self.load_model()
        
        # 2. 후크 등록
        self.register_hooks()
        
        # 3. 각 입력 텍스트에 대해 개별적으로 분석
        for i, text in enumerate(input_texts):
            print(f"\n\n처리 중인 텍스트 #{i+1}: {text[:50]}...")
            
            # 3.1. 텍스트로 모델 실행
            self.run_inference(text)
            
            # 3.2. 각 컴포넌트에 대한 시각화 생성
            print("\nGenerating visualizations...")
            for name in self.activation_stats:
                if "_input" in name:  # 입력 텐서만 분석 (출력은 다음 레이어의 입력)
                    print(f"Visualizing {name}...")
                    self.visualize_flat_channel_heatmap(name, i)
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 4. README 생성
        print("\nGenerating README...")
        with open(os.path.join(self.output_dir, "README.md"), 'w') as f:
            f.write("# LLaMA 모델 채널 활성화 분석 결과\n\n")
            f.write(f"- **모델:** {self.model_name}\n")
            f.write(f"- **디바이스:** {self.device}\n")
            f.write(f"- **정밀도:** {self.precision}\n\n")
            
            f.write("## 분석 내용\n\n")
            f.write("- 채널-토큰 히트맵: 모든 채널과 토큰에 대한 절대값 분포\n")
            f.write("- 채널 그룹 히트맵: 채널을 1000개씩 그룹화하여 상세 분석\n")
            f.write("- 이상치 채널 분석: 평균보다 유의미하게 높은 값을 가진 채널 식별\n\n")
            
            f.write("## 좌표 설명\n\n")
            f.write("- X축: 채널 인덱스 (가로)\n")
            f.write("- Y축: 토큰 인덱스 (세로)\n")
            f.write("- Z축/컬러맵: 절대값 크기\n\n")
            
            f.write("## 입력 텍스트 설명\n\n")
            for i, text in enumerate(input_texts):
                f.write(f"### 텍스트 #{i+1}\n")
                f.write(f"```\n{text[:200]}{'...' if len(text) > 200 else ''}\n```\n\n")
            
            f.write("## 디렉토리 구조\n\n")
            f.write("```\n")
            f.write(f"{self.output_dir}/\n")
            f.write("├── text_1/                  # 첫 번째 입력 텍스트\n")
            f.write("│   ├── layer_N/             # 레이어별 데이터\n")
            f.write("│   │   ├── attention_q_proj/    # 쿼리 투영\n")
            f.write("│   │   ├── attention_k_proj/    # 키 투영\n")
            f.write("│   │   ├── ...\n")
            f.write("├── text_2/                  # 두 번째 입력 텍스트\n")
            f.write("│   ├── ...\n")
            f.write("...\n")
            f.write("└── README.md                # 이 파일\n")
            f.write("```\n")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAnalysis complete. Took {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Visualizations saved to {self.output_dir}/")

# 사용 예시
if __name__ == "__main__":
    # 분석할 대상 모델
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # 분석기 초기화
    analyzer = ImprovedOutlierAnalyzer(
        model_name=model_name,
        device="cuda",
        precision="fp16",
        output_dir="magnitude_visualizations_wikitext"
    )
    
    # WikiText 데이터셋에서 5개 샘플 로드
    test_texts = load_wikitext_samples(num_samples=5, min_length=100, max_length=500)
    
    # 분석 실행
    analyzer.run_analysis(test_texts)