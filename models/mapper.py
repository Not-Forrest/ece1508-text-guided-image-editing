import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import CLIPTokenizer, CLIPTextModel
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# --- Cross Attention Module ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, txt_dim=512, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(txt_dim, dim)
        self.v_proj = nn.Linear(txt_dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.param_init()
    def param_init(self):
        # Initialize q_proj, k_proj, v_proj
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.kaiming_normal_(proj.weight, mode='fan_in', nonlinearity='relu')
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # Initialize Feed-Forward layers
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize LayerNorms
        for norm in [self.norm, self.norm2]:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)


    def forward(self, x, text, text_mask = None):
        # x: (B, N, dim), text: (B, L, txt_dim)
        q = self.q_proj(x)
        k = self.k_proj(text)
        v = self.v_proj(text)
        attn_out, _ = self.attn(q, k, v, key_padding_mask=text_mask)
        x = self.norm(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
    
class ConvMapper(nn.Module):
    def __init__(self, in_channels=256, text_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels + text_dim, 256, 3, padding=1),  
            nn.GroupNorm(8, 256),  
            nn.ReLU(),
            nn.Conv2d(256, in_channels, 1),
            nn.Tanh()  
        )

    def forward(self, u, z_txt):

        z_txt = z_txt.view(z_txt.size(0), z_txt.size(1), 1, 1)
        z_txt = z_txt.expand(-1, -1, u.size(2), u.size(3))


        attn_mask = self.attention(torch.cat([u, z_txt], dim=1))

        return u + 0.3 * attn_mask * u
    
class CAMapper(nn.Module):
    def __init__(self, in_channels=256, text_dim=512):
        super().__init__()
        self.attention = CrossAttentionBlock(in_channels)

    def forward(self, u, z_txt, num_layers = 4):
        B, C, H, W = u.shape
        u = u.view(B, C, H * W)
        u = u.permute(0, 2, 1)  # (B, H*W, C)
        z_txt = z_txt.unsqueeze(1) # (B, 1, text_dim)
        for _ in range(num_layers):
            u = self.attention(u, z_txt)
        u = u.permute(0, 2, 1)  # (B, C, H*W)
        u = u.view(B, C, H, W)
        return u
    
class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=DEVICE,
                 project_dim=None, prompt_engineering=True):
        """
        Args:
            project_dim: 投影到指定维度（None表示保持原维度）
        """
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.output_dim = self.model.config.hidden_size  # 512
        self.prompt_engineering = prompt_engineering

        # optional projection layer
        self.projector = nn.Linear(self.output_dim, project_dim) if project_dim else None
        self.final_dim = project_dim or self.output_dim

        self.action_templates = [
            "Modify the image to {}",
            "Apply the edit: {}",
            "Perform this transformation: {}",
            "Change the appearance to {}",
            "Alter the image by {}",
            "Implement the edit: {}",
            "Transform the subject to {}",
            "Execute the change: {}"
        ]

        # 动作强化副词
        self.action_adverbs = [
            "clearly", "visibly", "distinctly", "noticeably",
            "prominently", "significantly", "dramatically", "effectively"
        ]

        # 目标对象强化词
        self.target_descriptors = [
            "the subject", "the person", "the main focus",
            "the central figure", "the foreground"
        ]
    def enhance_action_prompts(self, sentences):
        """强化动作类提示词"""
        if not self.prompt_engineering:
            return sentences

        processed_sentences = []
        for text in sentences:
            # 检测是否包含动作关键词
            is_action = any(keyword in text.lower() for keyword in ["add", "remove", "change", "replace", "modify"])

            if is_action:
                # 1. 选择动作模板
                template = random.choice(self.action_templates)
                processed = template.format(text)

                # 2. 添加动作强化副词 (70%概率)
                if random.random() < 0.7:
                    adverb = random.choice(self.action_adverbs)
                    # 在动词后插入副词 (e.g. "add clearly blonde hair")
                    if "add" in processed:
                        processed = processed.replace("add", f"add {adverb}")
                    elif "remove" in processed:
                        processed = processed.replace("remove", f"remove {adverb}")
                    # 通用处理
                    processed = f"{adverb} {processed}"

                # 3. 添加目标对象强化 (50%概率)
                if random.random() < 0.5:
                    target = random.choice(self.target_descriptors)
                    processed = processed.replace("the", target, 1)  # 替换第一个"the"

                # 4. 添加编辑效果强化 (30%概率)
                if random.random() < 0.3:
                    effect = random.choice(["seamlessly", "naturally", "realistically", "coherently"])
                    processed += f" {effect}"
            else:
                processed = text  # 非动作提示保持原样

            processed_sentences.append(processed)

        return processed_sentences

    def forward(self, sentences):
        """返回文本特征表示

        Args:
            sentences: 字符串列表，长度B

        Returns:
            如果use_pooled=True: (B, D) 全局文本特征
            如果use_pooled=False: (B, L, D) 序列特征 + padding_mask
        """
        # add prompt engineering
        # print("before: ",sentences)
        if self.prompt_engineering:
            sentences = self.enhance_action_prompts(sentences)
        # print("after prompt e:",sentences)
        # Tokenize
        tokens = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP标准长度
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        # 提取特征
        with torch.no_grad():  # 冻结CLIP权重
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)


            # 使用[EOS]位置的特征作为全局表示
            last_token_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(sentences), device=self.device)
            z_txt = outputs.last_hidden_state[batch_indices, last_token_positions]

        # 可选维度投影
        if self.projector:
            z_txt = self.projector(z_txt)

        return z_txt
