import torch
import timm
import torch.nn as nn
from transformers.models.deberta_v2 import DebertaV2Model
import transformers

class AttentionPooling(nn.Module):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0, :] 

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
        sum_mask = attention_mask.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
           
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).float()
        x = last_hidden_state * attention_mask
        x = torch.max(x, dim=1)[0]
        return x        
        
class GemPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GemPooling, self).__init__()
        
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        
    def forward(self, last_hidden_state, attention_mask):
        
        last_hidden_state = last_hidden_state.clamp(min=self.eps).pow(self.p)
        
        attention_mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
        sum_mask = attention_mask.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        mean_embeddings = mean_embeddings.pow(1./self.p)
        
        return mean_embeddings  
      
class Stage1(torch.nn.Module):

    def __init__(self,
                 encoder_name='convnext_base.dinov3_lvd1689m',
                 pretrained=True,
                 classifier_dropout=0.0,
                 gc=False):
        
        super().__init__()
        
        self.image_encoder = timm.create_model(encoder_name,
                                               pretrained=pretrained,
                                               num_classes=0) 
        
        if gc:                                  
            self.image_encoder.set_grad_checkpointing(True)  
                                                        
        self.feature_info = self.image_encoder.feature_info       

        n = self.feature_info[-1]['num_chs']

        self.fc = torch.nn.Sequential(torch.nn.Dropout(classifier_dropout),
                                      torch.nn.Linear(n, 14)
                                      )
                                        
    def forward(self, x):
        
        f = self.image_encoder(x)
        x = self.fc(f)
        
        return x, f



class Stage2(torch.nn.Module):

    def __init__(self,
                 transformer_name="microsoft/deberta-v3-base",
                 hidden_size=1024,
                 intermediate_size=1014,
                 attention_heads=8,
                 num_hidden_layers=3,
                 attention_dropout=0.1,
                 hidden_dropout=0.1,
                 classifier_dropout=0.1,
                 gc=True,
                 pool="gem",
                 ):
        
        super().__init__()
        
        config = transformers.AutoConfig.from_pretrained(transformer_name)
        
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.vocab_size = 3
        config.num_hidden_layers = num_hidden_layers
        config.num_labels = 2
        config.num_attention_heads = attention_heads
        config.attention_probs_dropout_prob = attention_dropout
        config.hidden_dropout_prob = hidden_dropout
        config.hidden_act = "relu"
        
        config.conv_act = "relu"
        config.conv_kernel_size = 3
        
        print(config)
        
        self.config = config
        
        self.transformer = transformers.AutoModel.from_config(config)  
        
        if gc:
            self.transformer.gradient_checkpointing_enable()
    
        scale = hidden_size ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
    
        n = hidden_size  
        self.fc = torch.nn.Sequential(
                                  torch.nn.LayerNorm(n*2),
                                  torch.nn.Linear(hidden_size*2, n),
                                  torch.nn.LayerNorm(n),
                                  torch.nn.GELU(),
                                  torch.nn.Dropout(classifier_dropout),
                                  torch.nn.Linear(n, 14)
                                  )  
                                        
        if pool == "cls":
            self.pool = AttentionPooling()  
        elif pool == "gem":
            self.pool = GemPooling()
        elif pool =="max":
            self.pool = MaxPooling()
        else:
            self.pool = MeanPooling()   

    def forward(self, x, a):
        
        b, t, c = x.shape

        cls_emb = self.cls_embedding.repeat(b, 1, 1)
               
        a_add = torch.ones((b, 1), dtype=a.dtype, device=a.device)
        
        x = torch.cat([cls_emb, x], dim=1)
        a = torch.cat([a_add, a], dim=1)
        
        output = self.transformer(inputs_embeds=x, attention_mask=a)
        
        x = output.last_hidden_state
        
        x = torch.cat([self.pool(x, a), x[:, 0, :]], dim=-1)
        
        x = self.fc(x)
         
        return x
    
    
class Model(torch.nn.Module):

    def __init__(self,
                 encoder_name,
                 transformer_config,
                 classifier_dropout=0.0,
                 pool="gem",
                 ):
        
        super().__init__()
        
        self.image_encoder = timm.create_model(encoder_name,
                                               pretrained=False,
                                               num_classes=0) 
                                                                                      
        transformer_config._output_attentions = False
        transformer_config.output_attentions = False

        self.transformer = DebertaV2Model(transformer_config)

        hidden_size = transformer_config.hidden_size
        
        scale = hidden_size ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, hidden_size))
        
        n = hidden_size  
        
        self.fc = torch.nn.Sequential(
                                  torch.nn.LayerNorm(n*2),
                                  torch.nn.Linear(hidden_size*2, n),
                                  torch.nn.LayerNorm(n),
                                  torch.nn.GELU(),
                                  torch.nn.Dropout(classifier_dropout),
                                  torch.nn.Linear(n, 14)
                                  )  
                                        
        if pool == "cls":
            self.pool = AttentionPooling()  
        elif pool == "gem":
            self.pool = GemPooling()
        elif pool =="max":
            self.pool = MaxPooling()
        else:
            self.pool = MeanPooling()
      
    def forward_encoder(self, x):
        
        f = self.image_encoder(x)
        
        return f             
      
        
    def forward_transformer(self, x, a):
        
        b, t, c = x.shape

        cls_emb = self.cls_embedding.repeat(b, 1, 1)
               
        a_add = torch.ones((b, 1), dtype=a.dtype, device=a.device)
        
        x = torch.cat([cls_emb, x], dim=1)
        a = torch.cat([a_add, a], dim=1)
        
        output = self.transformer(inputs_embeds=x, attention_mask=a)
        
        x = output.last_hidden_state
        
        x = torch.cat([self.pool(x, a), x[:, 0, :]], dim=-1)
        
        x = self.fc(x)

        return x 