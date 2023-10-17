import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention_Layer(nn.Module):
    def __init__(self, embedding_dim, heads = 8, dim_head = 128):
        super().__init__()

        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.Norm_Layer = nn.LayerNorm(embedding_dim)

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, input):
        # print(f"input shape is {input.shape}")
        x = self.Norm_Layer(input)

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = self.attend(torch.matmul(q, k.transpose(-2, -1)) * self.scale) # [b, h, n, n]

        output = torch.matmul(attn, v) # [b, h, n, d]
        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.to_out(output) + input

class Trans_MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim) -> None:
        super().__init__()
        self.Norm_Layer = nn.LayerNorm(embedding_dim)
        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, input):
        return self.MLP(self.Norm_Layer(input)) + input
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, hidden_dim) -> None:
        super().__init__()
        modules = []
        for i in range(depth):
            modules.append(Attention_Layer(embedding_dim=embedding_dim))
            modules.append(Trans_MLP(embedding_dim, hidden_dim=hidden_dim))
        
        self.models = nn.Sequential(*modules)
    def forward(self, x):
        return self.models(x)

def patchify(input, patch_size):
    """
    input  = [batch, channel, height, width]
    output = [batch, num_patches, patch_size**2*channel]
    """
    batch_size, channel, image_size, _ = input.shape
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    num_patches = (image_height // patch_height) * (image_width // patch_width)
    patch_dim = patch_height * patch_width * channel

    re_arrange = Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = patch_height, p2 = patch_width)

    return re_arrange(input), num_patches, patch_dim

class MAE(nn.Module):
    def __init__(self, image_size, patch_size, depth=6, hidden_dim=64, embedding_dim=1024, decoder_embed_dim=512):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = patch_height * patch_width
        self.patch_size = patch_size
        
        self.get_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(self.patch_dim, embedding_dim)
        )
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim), requires_grad=False)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.Encoder = Transformer(embedding_dim=embedding_dim, depth=depth, hidden_dim=hidden_dim)

        self.decoder_embed = nn.Sequential(
                nn.Linear(embedding_dim, decoder_embed_dim, bias=True),
                nn.Dropout(0.1)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2, bias=True) # decoder to patch
        
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # print(f"len_keep is :{len_keep}")
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # print(f"noise shape is :{noise.shape}")

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print(f"ids_shuffle shape is :{ids_shuffle.shape}")
        # print(f"ids_shuffle is :{ids_shuffle}")
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print(f"ids_restore shape is :{ids_restore.shape}")
        # print(f"ids_restore is :{ids_restore}")
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # print(f"ids_keep is :{ids_keep}")
        # print(f"index is {ids_keep.unsqueeze(-1).repeat(1, 1, D)}")
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # print(f"x_masked shape is :{x_masked.shape}")
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # print(f"mask shape is {mask.shape}")

        return x_masked, mask, ids_restore

    def decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1) # [1, 1, embedding_dim] => [batch_size, residual patches, embedding_dim]
        # print(f"mask_tokens shape is :{mask_tokens.shape}")
        x = torch.cat([x, mask_tokens], dim=1)
        # print(f"after cat, x shape is {x.shape}")
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # print(f"after unshuffle, x shape is {x.shape}")

        x = x + self.decoder_pos_embed
        
        return self.decoder_pred(x)
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, H, W]
        pred: [N, L, p*p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        reshaper = Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = self.patch_size, p2 = self.patch_size)
        target = reshaper(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, image, mask_ratio):
        x = self.get_patch_embedding(image) # [batch, num_patches, embedding_dim]
        batch, num_patches, embedding_dim = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos
        # mask the data
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        x = self.Encoder(x) # [batch, num_patches - mask, embedding]
        class_feature = x

        pred_pic = self.decoder(x, ids_restore)

        loss = self.forward_loss(image, pred_pic, mask)

        return loss, pred_pic, mask, class_feature
    
class Ensemble(nn.Module):
    def __init__(self, backbone, out_channels, gender_dim) -> None:
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, gender_dim),
            nn.BatchNorm1d(gender_dim),
            nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(256 + gender_dim, 240),
            nn.Softmax(dim=1)
        )
    

    def forward(self, image, gender, mask_ratio):
        loss, pred_pic, mask, x = self.backbone(image, mask_ratio)
        batch, patches, dim = x.shape
        image_size = int(dim**.5)
        x = torch.reshape(x, [batch, patches, image_size, image_size])
        x = self.conv2(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.squeeze(x)
        x = x.view(-1, 256)

        gender_encode = self.gender_encoder(gender)

        x = torch.cat([x, gender_encode], dim=1)
        # print(x.shape)
        return loss, pred_pic, mask, self.MLP(x)
    
    def fine_tune(self, isClass):
        notClass = not isClass
        for param in self.backbone.parameters():
            param.requires_grad = notClass
        for param in self.gender_encoder.parameters():
            param.requires_grad = isClass
        for param in self.MLP.parameters():
            param.requires_grad = isClass
        for param in self.conv2.parameters():
            param.requires_grad = isClass            
            
if __name__ == '__main__':
    loss_fn = nn.L1Loss(reduction='sum')
    image = torch.randint(0, 10, (10 ,512, 512), dtype=torch.float).cuda()
    num_hiddens = 512
    gender = torch.randint(0, 2, (10, 1), dtype=torch.float).cuda()
    mask_ratio = 0.75
    embed_dim = 1024
    net = MAE(image_size=512, patch_size=32, depth=6, hidden_dim=num_hiddens, embedding_dim=embed_dim)
    net = Ensemble(net, embed_dim, 32)
    # print(net)

    net = net.cuda()
    # net(image, gender, mask_ratio)
    net.fine_tune(True)
