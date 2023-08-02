import os
import torch
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.libero.benchmark import get_benchmark
from libero.libero import get_libero_path
from omegaconf import OmegaConf
from arch import MineCLIP
from sklearn.metrics.pairwise import cosine_similarity

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

    
    print('======== DATASET INFORMATION ========')
    print('Number of tasks:', len(manip_datasets))

    return descriptions, manip_datasets
    

@torch.no_grad()
@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OmegaConf.set_struct(cfg, False)
    cfg.pop("ckpt")
    cfg.pop("wandb")
    OmegaConf.set_struct(cfg, True)

    model = MineCLIP(**cfg).to(device)
    model.load_state_dict(torch.load('model/model_spatial.pth')) # Load model weights

    prompts, manip_datasets = load_dataset()
    videos = torch.stack([torch.tensor(video[0]['obs']['agentview_rgb']) for video in manip_datasets]).to(device) # Select all the videos

    # Plot embedding similarity
    vid_embeddings = model.encode_video(videos).cpu().numpy()
    text_embeddings = model.encode_text(prompts).cpu().numpy()

    similarity_matrix = cosine_similarity(vid_embeddings)
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Video Embeddings Cosine Similarity Matrix')
    plt.xlabel('Embedding Index')
    plt.ylabel('Embedding Index')
    plt.savefig('graphs/vid_embd.pdf')
    plt.clf()
    
    similarity_matrix = cosine_similarity(text_embeddings)
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Text Embeddings Cosine Similarity Matrix')
    plt.xlabel('Embedding Index')
    plt.ylabel('Embedding Index')
    plt.savefig('graphs/text_embd.pdfç')

    # Compute rewards
    reward, _ = model(
        videos, text_tokens=prompts, is_video_features=False
    )
    print("----Rewards----")
    print(reward)

if __name__ == "__main__":
    main()
