import os
import torch
import hydra
import pytorch_lightning as pl
import torch.optim as optim
import wandb
import yaml

from arch import MineCLIP 
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.libero.benchmark import get_benchmark
from libero.libero import get_libero_path
from torchinfo import summary


def load_dataset():
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
                seq_len=16,
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
    
    print('======== DATASET INFORMATION ========')
    print('Number of tasks:', len(datasets))
    print('Number of demontrations per task:', [ds.n_demos for ds in datasets])

    concat_dataset = ConcatDataset(datasets)
    return concat_dataset

class MineCLIPSystem(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def train_dataloader(self):
        dataset = load_dataset()
        train_dataloader = DataLoader(dataset, batch_size=64, sampler=RandomSampler(dataset)) 
        return train_dataloader

    def freeze_layers(self):
        for name, param in self.model.clip_model.vision_model.named_parameters():
            if name.startswith("blocks.11"):
                break
            param.requires_grad = False
            
        for name, param in self.model.clip_model.text_model.named_parameters():
            if name.startswith("blocks.11"):
                break
            param.requires_grad = False 

    def forward(self, obs, text):        
        logits_per_video, logits_per_text = self.model(obs, text)
        return logits_per_video, logits_per_text
    
    def training_step(self, batch, batch_idx):
        obs, text = batch["obs"]["agentview_rgb"], batch["task_emb"]
        logits_per_video, logits_per_text = self.forward(obs, text)

        # INFOSCE OBJECTIVE
        # batch_size = logits_per_video.size(0)
        # pos_logits = torch.diag(logits_per_video) + torch.diag(logits_per_text)
        # full_logits = logits_per_video + logits_per_text
        # neg_logits = full_logits - 2 * torch.diag(torch.diag(full_logits))
        # neg_sum = torch.logsumexp(neg_logits, dim=1)
        # loss = - torch.mean(pos_logits - neg_sum)

        labels = torch.arange(logits_per_video.shape[0], device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        image_loss = loss_fn(logits_per_video, labels)  
        text_loss = loss_fn(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2

        self.log("train_loss", loss, prog_bar=True)
        self.model.clamp_logit_scale()
        return loss
    
    def configure_optimizers(self):
        self.freeze_layers() # Freeze pre-trained layers

        groups = [self.model.temporal_encoder, self.model.reward_head] # Reward head contains image encoder and text model parameters
        peak_lr = 1.5e-4
        final_lr = 1e-5
        weight_decay=0.2
        decay = 0.65
        
        dataset_size = len(self.train_dataloader().dataset)
        batch_size = self.train_dataloader().batch_size
        num_steps_per_epoch = dataset_size // batch_size
        total_steps = 2 * num_steps_per_epoch

        normal_param_group = []
        clip_param_group = []
        layer_param_group = []

        for group in groups:
            layers = list(reversed(list(group.named_parameters())))
            for (name, param) in layers:
                if name.startswith('clip_model'):
                    if name.startswith('clip_model.vision_model.blocks.11') or name.startswith('clip_model.text_model.blocks.11'):
                        layer_param_group.append(param)
                    else:
                        clip_param_group.append(param)
                else:
                    normal_param_group.append(param)

        param_groups = [
            {'params': normal_param_group, 'lr': peak_lr},
            {'params': clip_param_group, 'lr': peak_lr * 0.5},
            {'params': layer_param_group, 'lr': peak_lr * 0.5 * decay},
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=final_lr) # NOTE: Not using decay on final learning rate.

        # NOTE: Warmup not implemented.

        return ([optimizer], [{'scheduler': scheduler, 'interval': 'step'}])


@hydra.main(config_name="conf", config_path="main/", version_base="1.1")
def train(cfg):
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    wandb_config = cfg.pop("wandb")
    OmegaConf.set_struct(cfg, True)

    model = MineCLIP(**cfg)
    model.load_ckpt(ckpt.path, strict=False)
    system = MineCLIPSystem(model, cfg)
    wandb_logger = WandbLogger(project=wandb_config.project_name, name=wandb_config.run_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # View layers of the MineCLIP Model. NOTE: Must change 'text' for test.
    # system.configure_optimizers()
    # summary(system.model, input_size=(16, 16, 3, 128, 128), mode='train', device = device, verbose=1, col_names=["trainable", "input_size", "output_size", "num_params"],)

    trainer = Trainer(logger=wandb_logger, callbacks=[lr_monitor], max_epochs=1, log_every_n_steps=1, strategy=DDPStrategy(find_unused_parameters=True)) 
    trainer.fit(system)
    torch.save(system.model.state_dict(), '../model/model_spatial.pth')

if __name__ == "__main__":
    train()
