import torch
import os
import hydra
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from arch import MineCLIP
from warmup_scheduler import GradualWarmupScheduler

from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.libero.benchmark import get_benchmark
from libero.libero import get_libero_path
from torchsummary import summary

def load_dataset():
    """
    Load dataset here.
    """
    name = "libero_spatial"
    datasets_default_path = get_libero_path("datasets")

    benchmark = get_benchmark(name)(0)
    tasks = benchmark.n_tasks

    manip_datasets = []
    descriptions = []

    for i in range(tasks):
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    datasets_default_path, benchmark.get_task_demonstration(i)
                ),
                obs_modality={'rgb': ['agentview_rgb']},
                initialize_obs_utils=(i == 0),
                seq_len=10,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    datasets = [
        SequenceVLDataset(ds, descrip) for (ds, descrip) in zip(manip_datasets, descriptions)
    ]
    concat_dataset = ConcatDataset([ds for ds in datasets]) # TODO: Figure out the right way to concatenate datasets.
    
    print('======== DATASET INFORMATION ========')
    print('Number of tasks:', len(concat_dataset))
    print('Number of demontrations per task:', [ds.n_demos for ds in datasets])

    return concat_dataset

@hydra.main(config_name="conf", config_path="main/", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)
    model = MineCLIP(**cfg).to(device)
    # model.load_ckpt(ckpt.path, strict=False) # Load CLIP checkpoint.
    model.train()

    for name, module in model.named_children():
        print(name)

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

    TODO: Similar as above. Weird configuration.
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
