import torch
import hydra
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from arch import MineCLIP
from warmup_scheduler import GradualWarmupScheduler

def load_dataset():
    """
    Load dataset here. 

    They sample 640K pairs of 16-second video snippets + time-aligned English transcripts:
    1. They gather a list of popular keywords in Minecraft and search through transcripts to find 640K text segments that match.
    2. They randomly "grow" their text segement to 16 ~ 77 tokens by adding words before and after.
    3. They randomly sample a timestep within their text-video segment and grow it to 8 ~ 16 seconds.
    4. They sample 16 RGB frames from their video segment uniformly.

    Lastly, they apply data agumentation via a temporally-consistent random resized crop.
    """
    dataset = VideoDataset(videos)
    return videos

@hydra.main(config_name="conf", config_path="main/", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt") # Set as CLIP checkpoint path.
    OmegaConf.set_struct(cfg, True)
    model = MineCLIP(**cfg).to(device)
    # model.load_ckpt(ckpt.path, strict=True)
    model.train()

    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True) # Batch size of 64 / GPU.

    """
    Freeze image encoder and text encoder except for final 2 layers.

    TODO: Paper and code do not align for image and text encoder placement. Also, check if this works.
    """
    for child in list(model.image_encoder.children())[:-2]:
        for param in child.parameters():
            param.requires_grad = False
    for child in list(model.clip_model.text_model.children())[:-2]:
        for param in child.parameters():
            param.requires_grad = False

    """
    Pre-trained layers get 0.5x learning rate multiplier. We also have a 0.65 layer learning rate decay. 

    TODO: Similar as above. Weird configuration, currently treating reward_head as text_model. Also, check if this works :)
    """
    parts = [model.image_encoder, model.temporal_encoder, model.reward_head]
    base_lr = 1.5e-4
    decay = 0.65
    params = []
    for part in parts:
        layers = list(part.children())
        if part == model.image_encoder or part == model.reward_head:
            lr = base_lr / 2
        else:
            lr = base_lr
        for i, layer in enumerate(layers):
            params.append({'params': layer.parameters(), 'lr': lr * (decay ** i)})

    optimizer = optim.AdamW(params, weight_decay=0.2) # Which optimizer do they use?
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5) 
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=500, after_scheduler=scheduler_cosine)

    for epoch in range(2): # Run for 2 epochs.
        for batch in dataloader:
            video, text = batch["video"].to(device), batch["text"]
            
            optimizer.zero_grad()

            video_features = model.encode_video(video)
            text_tokens = model.encode_text(text)
            logits_per_video, logits_per_text = model.forward_reward_head(video_features, text_tokens)

            # InfoNCE loss... Do we include negative pairs?
            logit_scale = model.clip_model.logit_scale.exp()
            sim_matrix = logit_scale * logits_per_video @ logits_per_text.t()
            loss = (-torch.diag(F.log_softmax(sim_matrix, dim=-1))).mean()


            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Finished epoch {epoch+1} with loss {loss.item()}")

if __name__ == "__main__":
    main()
