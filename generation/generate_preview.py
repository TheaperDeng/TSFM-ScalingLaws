import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Union, List

# Add TimeCraft/TimeDP to path
sys.path.append("/home/v-junweideng/online-ts-aug/TimeCraft/TimeDP")


def generate_conditional_batch(
    prompt: Union[np.ndarray, torch.Tensor, List[float]],
    subset_id,
    model,
    data_module,
    dataset_name: str,
    dataset_idx: int,
    num_samples: int = 1,
    ddim: bool = True,
    ddim_steps: int = 40,
    device: str = 'cuda',
    context_length: int = 7,  # Number of observed timesteps
    prediction_length: int = 3,  # Number of prediction timesteps  
    total_length: int = 512,  # Total sequence length
    patch_size: int = 32,  # Patch size
    sample_id: int = 1
) -> dict:
    """
    Generate conditional samples and wrap them in the expected batch dictionary format.
    
    Args:
        prompt: Input prompt as numpy array, torch tensor, or list
        model: The loaded TimeDP model
        data_module: The data module for normalization
        dataset_name: Name of the dataset for normalization
        dataset_idx: Dataset index for metadata
        num_samples: Number of samples to generate (batch size)
        ddim: Whether to use DDIM sampling
        ddim_steps: Number of DDIM steps
        device: Device to run on
        context_length: Number of observed timesteps (183 in example)
        prediction_length: Number of prediction timesteps (64 in example)
        total_length: Total sequence length (512)
        patch_size: Patch size (32)
        sample_id: Sample ID for metadata
        
    Returns:
        Dictionary with keys: target, observed_mask, time_id, variate_id, prediction_mask,
        patch_size, label, label_observed_mask, _dataset_idx, sample_id, dataset_index
    """
    
    # Generate samples using the existing function
    generated_samples = generate_conditional_samples(
        prompt=prompt,
        subset_id=subset_id,
        model=model,
        data_module=data_module,
        dataset_name=dataset_name,
        num_samples=num_samples,
        ddim=ddim,
        ddim_steps=ddim_steps,
        device=device
    )
    
    # Convert to torch tensors if needed
    if isinstance(generated_samples, np.ndarray):
        generated_samples = torch.tensor(generated_samples, dtype=torch.float32)
    
    # Ensure we have the right shape: (num_samples, sequence_length)
    if generated_samples.dim() == 1:
        generated_samples = generated_samples.unsqueeze(0)
    
    batch_size = generated_samples.shape[0]
    sequence_length = generated_samples.shape[1]
    
    # Reshape generated samples to (batch_size, total_length, patch_size)
    # We need to pad or truncate to fit the expected format
    if sequence_length < total_length * patch_size:
        # Pad with zeros
        padding_needed = total_length * patch_size - sequence_length
        padding = torch.zeros(batch_size, padding_needed, dtype=torch.float32)
        padded_samples = torch.cat([generated_samples, padding], dim=1)
    else:
        # Truncate to fit
        padded_samples = generated_samples[:, :total_length * patch_size]
    
    # Reshape to (batch_size, total_length, patch_size)
    target = padded_samples.view(batch_size, total_length, patch_size)
    
    # Create observed_mask: True for observed timesteps, False for padding
    observed_mask = torch.zeros(batch_size, total_length, patch_size, dtype=torch.bool)
    observed_timesteps = min(context_length, total_length)
    observed_mask[:, :observed_timesteps, :] = True
    
    # Create time_id: sequential timestep IDs
    time_id = torch.zeros(batch_size, total_length, dtype=torch.long)
    for i in range(batch_size):
        time_id[i, :observed_timesteps] = torch.arange(observed_timesteps)
        # Padding timesteps get 0
    
    # Create variate_id: all zeros (univariate)
    variate_id = torch.zeros(batch_size, total_length, dtype=torch.long)
    
    # Create prediction_mask: True for prediction timesteps
    prediction_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    pred_start = context_length
    pred_end = min(context_length + prediction_length, total_length)
    prediction_mask[:, pred_start:pred_end] = True
    
    # Create patch_size tensor
    patch_size_tensor = torch.full((batch_size, total_length), patch_size, dtype=torch.long)
    # Set padding areas to 0
    patch_size_tensor[:, observed_timesteps:] = 0
    
    # Label is the same as target for this case
    label = target.clone()
    
    # Label observed mask is the same as observed mask
    label_observed_mask = observed_mask.clone()
    
    # Dataset index tensor
    _dataset_idx = torch.tensor([dataset_idx] * batch_size, dtype=torch.long)
    
    # Sample ID tensor
    sample_id_tensor = torch.full((batch_size, total_length), sample_id, dtype=torch.long)
    # Set padding areas to 0
    sample_id_tensor[:, observed_timesteps:] = 0
    
    # Dataset index for each timestep
    dataset_index = torch.full((batch_size, total_length), dataset_idx, dtype=torch.long)
    # Set padding areas to -1
    dataset_index[:, observed_timesteps:] = -1
    
    # Create the batch dictionary
    batch = {
        'target': target,
        'observed_mask': observed_mask,
        'time_id': time_id,
        'variate_id': variate_id,
        'prediction_mask': prediction_mask,
        'patch_size': patch_size_tensor,
        'label': label,
        'label_observed_mask': label_observed_mask,
        '_dataset_idx': _dataset_idx,
        'sample_id': sample_id_tensor,
        'dataset_index': dataset_index
    }
    
    return batch


def generate_conditional_samples(
    prompt: Union[np.ndarray, torch.Tensor, List[float]],
    subset_id,
    model,
    data_module,
    dataset_name: str = None,
    num_samples: int = 1,
    ddim: bool = True,
    ddim_steps: int = 20,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Generate conditional samples using a trained TimeDP model.
    
    Args:
        prompt: Input prompt as numpy array, torch tensor, or list. Shape should be (sequence_length,)
        model: The loaded TimeDP model
        data_module: The data module for normalization
        dataset_name: Name of the dataset for normalization (if None, uses first available dataset)
        num_samples: Number of samples to generate
        ddim_steps: Number of DDIM steps
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Generated samples as numpy array of shape (num_samples, sequence_length)
    """
    # Convert prompt to numpy array if needed
    if isinstance(prompt, torch.Tensor):
        prompt = prompt.detach().cpu().numpy()
    elif isinstance(prompt, list):
        prompt = np.array(prompt)
    
    # Ensure prompt is the right shape
    if prompt.ndim == 1:
        prompt = prompt.reshape(1, 1, -1)  # (1, 1, sequence_length)
    elif prompt.ndim == 2:
        prompt = prompt.reshape(prompt.shape[0], 1, -1)  # (batch, 1, sequence_length)
    
    # if dataset_name:
    normalizer = data_module.normalizer_dict[dataset_name]
    normalized_prompt = data_module.transform(prompt, normalizer)
    
    # Convert to tensor and move to device
    x = torch.tensor(normalized_prompt).to(device).float()
    if subset_id is not None:
        subset_id = torch.tensor([subset_id] * num_samples, dtype=torch.long).to(device)  # Dummy subset_id for now
    else:
        subset_id = None
    
    # Get conditioning
    c, mask = model.get_learned_conditioning(x, return_mask=True)

    # print("c.shape", c.shape if c is not None else "None")
    # print("mask.shape", mask.shape if mask is not None else "None")
    
    # Repeat conditioning for the number of samples we want to generate
    if c is None:
        mask_repeat = None
        cond = None
    elif mask is None:
        cond = c.repeat(num_samples, 1, 1)
        mask_repeat = None
    else:
        cond = c.repeat(num_samples, 1, 1)
        mask_repeat = mask.repeat(num_samples, 1)
    
    # Generate samples
    with torch.no_grad():
        samples, _ = model.sample_log(
            cond=cond, 
            batch_size=num_samples, 
            ddim=ddim, 
            ddim_steps=ddim_steps, 
            cfg_scale=5.0,  # 5.0 
            mask=mask_repeat,
            data_key=subset_id,
        )
        norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
        inv_samples = data_module.inverse_transform(norm_samples, data_name=dataset_name)

    return inv_samples.squeeze()


def load_timedp_model(
    config_path: str,
    ckpt_path: str,
    seq_len: int = 320,
    num_latents: int = 16,
    batch_size: int = 16,
    use_pam: bool = True,
    uncond: bool = False,
    seed: int = 0,
    debug: bool = False,
    overwrite_learning_rate: float = None,
    device: str = 'cuda'
):
    """
    Load TimeDP model and data module without using parser.
    
    Args:
        config_path: Path to the config YAML file
        ckpt_path: Path to the model checkpoint
        seq_len: Sequence length (default: 320)
        num_latents: Number of latents for PAM (default: 16)
        batch_size: Batch size (default: 16)
        use_pam: Whether to use PAM (default: True)
        uncond: Whether to use unconditional model (default: False)
        seed: Random seed (default: 0)
        debug: Debug mode (default: False)
        overwrite_learning_rate: Override learning rate (optional)
        device: Device to load on (default: 'cuda')
        
    Returns:
        Tuple of (model, data, config_name, logdir)
    """
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    from ldm.util import instantiate_from_config
    
    # Get data root from environment
    data_root = os.environ['DATA_ROOT']
    
    # Set seed
    seed_everything(seed)
    
    # Load and merge configs
    config = OmegaConf.load(config_path)
    
    # Get config name from path
    cfg_fname = os.path.split(config_path)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    
    # Customize config from parameters
    config.model['params']['seq_len'] = seq_len
    config.model['params']['unet_config']['params']['seq_len'] = seq_len
    config.data['params']['window'] = seq_len
    config.data['params']['batch_size'] = batch_size
    
    # Set max steps
    config.lightning['trainer']['max_steps'] = 50000
    if debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    
    # Handle learning rate
    if overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = overwrite_learning_rate
        print(f"Setting learning rate (overwriting config file) to {overwrite_learning_rate:.2e}")
        base_lr = overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
    
    # Create experiment name
    nowname = f"{cfg_name.split('-')[-1]}_{seq_len}_nl_{num_latents}_lr{base_lr:.1e}_bs{batch_size}"
    
    # Configure conditional/unconditional setup
    if uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = seq_len
        
        if use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            nowname += f"_pam"
        else:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False
    
    nowname += f"_seed{seed}"
    logdir = os.path.join('./logs', cfg_name, nowname)
    
    # Set checkpoint path in config
    config.model['params']['ckpt_path'] = ckpt_path
    
    # Instantiate model
    model = instantiate_from_config(config.model)
    
    # Instantiate data module
    # Replace placeholders in data paths
    for k, v in config.data.params.data_path_dict.items():
        config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(seq_len))
    
    data = instantiate_from_config(config.data)
    
    # Prepare data
    data.prepare_data()
    data.setup("predict")
    print("#### Data Preparation Finished #####")
    
    return model, data, cfg_name, logdir

def load_model(ckpt_path: str, config_path: str = None, device: str = 'cuda', seed: int = 0, **kwargs):
    """
    Load the TimeDP model and data module.
    
    Args:
        ckpt_path: Path to the model checkpoint
        config_path: Path to the config file (optional)
        device: Device to load on
        **kwargs: Additional arguments passed to load_timedp_model
        
    Returns:
        Tuple of (model, data)
    """
    if config_path is None:
        # config_path = '/home/v-junweideng/online-ts-aug/TimeCraft/TimeDP/configs/9_subdataset_timedp.yaml'
        config_path = '/home/v-junweideng/online-ts-aug/TimeCraft/TimeDP/configs/multi_domain_timedp.yaml'
    
    model, data, cfg_name, logdir = load_timedp_model(
        config_path=config_path,
        ckpt_path=ckpt_path,
        device=device,
        seed=seed,
        **kwargs
    )
    
    model.init_from_ckpt(ckpt_path)
    model = model.to(device)
    model.eval()
    
    return model, data

def main(model, data, dataset, index=0):
    """
    Test function for conditional generation.
    """

    # print(data.key_list)
    # for key, _ in data.data_dict.items():
    #     print(key)

    # load one from /home/v-junweideng/online-ts-data-opt/TSFM-ScalingLaws/analysis/extracted_label_patches_largest_2021.npy
    dataset_name = dataset  # "australian_electricity_demand" / "largest_2021"
    subset_id = data.key_list.index(dataset_name)
    # subset_id = 2
    print(dataset_name, subset_id, index)
    # test_prompt = np.load('/home/v-junweideng/online-ts-data-opt/TSFM-ScalingLaws/analysis/extracted_label_patches_largest_2021.npy')[10].reshape(320,)
    test_prompt = np.load(f'/home/v-junweideng/online-ts-data-opt/TSFM-ScalingLaws/analysis/extracted_label_patches_{dataset_name}.npy')[index].reshape(320,)
    
    print("Generating conditional samples...")
    print(f"Prompt shape: {test_prompt.shape}")
    
    # Generate samples
    generated_samples = generate_conditional_batch(
        prompt=test_prompt,
        subset_id=subset_id,
        model=model,
        dataset_name=dataset_name,
        data_module=data,
        num_samples=10,
        ddim=False,
        ddim_steps=40,
        dataset_idx=9494,
    )

    # print other information
    # print(generated_samples)
    
    print(f"Generated {len(generated_samples['label'])} samples")
    generated_samples = generated_samples['label'][:, :10, :].reshape(-1, 320)

    # plot all the generated samples in sub figures as well as the test prompt
    import matplotlib.pyplot as plt

    plt.subplots(len(generated_samples), 1, figsize=(10, 4 * len(generated_samples)))
    for i, sample in enumerate(generated_samples):
        plt.subplot(len(generated_samples), 1, i + 1)
        if i != 0:
            plt.plot(sample, label=f'Generated Sample {i+1}')
        if i == 0:  # Only plot the prompt on the first subplot
            plt.plot(test_prompt, label='Test Prompt', color="orange")
        plt.legend()
        plt.title(f'Generated Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(f'new_generated_samples_{dataset_name}_conditionclass_promotidx_{index}_eval.png')
    # plt.savefig(f'play.png')

if __name__ == '__main__':
    # Configuration
    # ckpt_path = "/home/v-junweideng/online-ts-data-opt/TimeCraft/TimeDP/logs/multi_domain_timedp/multi_domain_timedp_320_nl_16_lr1.0e-04_bs128_time1753544256.1366603_pam_seed0/checkpoints/000458-0.1169.ckpt"
    # ckpt_path = "/home/v-junweideng/online-ts-data-opt/TimeCraft/TimeDP/logs/multi_domain_timedp/multi_domain_timedp_320_nl_16_lr1.0e-04_bs128_time1753806424.1802049_pam_seed0/checkpoints/000043-0.1245.ckpt"
    # ckpt_path = "/home/v-junweideng/online-ts-aug/TimeCraft/TimeDP/logs/multi_domain_timedp/multi_domain_timedp_320_nl_16_lr1.0e-04_bs128_pam_seed0/checkpoints/000786-0.1152.ckpt"
    # ckpt_path = "/home/v-junweideng/online-ts-aug/TimeCraft/TimeDP/logs/multi_domain_timedp/multi_domain_timedp_320_nl_16_lr1.0e-04_bs128_time1753900355.8672683_pam_seed0/checkpoints/000060-0.0853.ckpt"
    ckpt_path = "000060-0.0853.ckpt"
    
    # Load model once
    print("Loading model...")
    model, data = load_model(ckpt_path, seed=0)
    model.eval()

    for dataset in data.key_list:
        print(f"Testing dataset: {dataset}")
        main(model, data, dataset, index=2)
