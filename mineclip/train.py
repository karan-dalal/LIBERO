import os
import torch
import hydra
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import re

from arch import MineCLIP 
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, ConcatDataset
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

    def freeze_layers(self):
        for name, param in self.model.clip_model.vision_model.named_parameters():
            if name.startswith("blocks.11"):
                break
            param.requires_grad = False
            
        for name, param in self.model.clip_model.text_model.named_parameters():
            if name.startswith("blocks.11"):
                break
            param.requires_grad = False 
    
    def training_step(self, batch, batch_idx):
        obs, text = batch["obs"]["agentview_rgb"], batch["task_emb"]
        logits_per_video, logits_per_text = self(obs, text)
        
        labels = torch.arange(logits_per_video.shape[0])
        loss_fn = torch.nn.CrossEntropyLoss()
        image_loss = loss_fn(logits_per_video, labels)  
        text_loss = loss_fn(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2

        self.log("train_loss", loss, prog_bar=True)
        self.clamp_logit_scale()
        return loss
    
    def configure_optimizers(self):
        self.freeze_layers() # Freeze pre-trained layers

        params = []
        groups = [self.model.clip_model.vision_model, self.model.clip_model.text_model, self.model.temporal_encoder, self.model.reward_head]
        base_lr = 1.5e-4
        decay = 0.65
        
        for i, group in enumerate(groups):
            layers = list(reversed(list(group.named_parameters())))
            lr = base_lr / 2 if i < 2 else base_lr # Half learning rate for pre-trained layers
            prev_layer_name = layers[0][0].split('.')[0]

            for idx, (name, param) in enumerate(layers):
                if i < 2: # Layerwise decay if image encoder or text model
                    curr_layer_name = name.split('.')[0] if name.split('.')[0] != 'blocks' else '.'.join(name.split('.')[:2])
                    if curr_layer_name != prev_layer_name:
                        lr *= decay
                        prev_layer_name = curr_layer_name   

                if param.requires_grad and not (name.startswith('clip_model') and i == 3): # Exclude reward head from including image encoder and text model parameters
                    # print(name, lr)
                    params.append({'params': param, 'lr': lr})
    
        optimizer = optim.AdamW(params, weight_decay=0.2)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=2)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=500, after_scheduler=scheduler_cosine)

        return ([optimizer], [scheduler])


@hydra.main(config_name="conf", config_path="main/", version_base="1.1")
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset()
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True) 

    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt.path, strict=False)
    system = MineCLIPSystem(model, cfg)
    logger = TensorBoardLogger('tensorboard', name='robo_clip')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # View layers of the MineCLIP Model. NOTE: Must change 'text' for test.
    # system.configure_optimizers()
    # summary(system.model, input_size=(16, 16, 3, 128, 128), mode='train', device = device, verbose=1, col_names=["trainable", "input_size", "output_size", "num_params"],)

    trainer = Trainer(logger=logger, callbacks=[lr_monitor], max_epochs=2, log_every_n_steps=0) 
    trainer.fit(system, train_dataloader)

if __name__ == "__main__":
    train()
