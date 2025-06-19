from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
import math

from tsfm.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
)
from tsfm.module.norm import RMSNorm
from tsfm.module.position import (
    LearnedEmbedding,
    LearnedProjection,
)
from tsfm.optim import SchedulerType, get_scheduler
from tsfm.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    EvalCrop_AdaLength,
    EvalPad_AdaLength,
    EvalMaskedPrediction,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .module import BasicModule
from tsfm.val.metrics import (
    MSE_mean,
    MAE_mean,
    MSE_median,
    MAE_median,
    MASE,
    MAPE,
    SMAPE,
    RMSE,
    NRMSE,
    ND,
    CRPS
)

class TransformerEncoderPretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
        "label",
        "label_observed_mask",
    )
    train_seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }
    
    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        num_training_steps: int,
        num_warmup_steps: int,
        max_dim: int = 1,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[BasicModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = [
            MSE_mean() ,MAE_mean(), MSE_median(), MAE_median(), MASE(), MAPE(), SMAPE(), RMSE(), NRMSE(), ND(), CRPS()
            ],
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        num_low_influence_to_remove: int = 16,
        enable_influence_scoring: bool = True,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = BasicModule(**module_kwargs) if module is None else module
        self.influence_scores = {}
        
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        output = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return output
    
    def infer(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        distr = self.forward(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            patch_size=patch_size,
            prediction_mask=prediction_mask,
        )
        
        preds = distr.sample(torch.Size((self.hparams.num_samples, ))) # sample batch time features
        preds = preds.transpose(0, 1) # batch sample time features
        return distr, preds

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Filter out low-influence samples from batch if influence scoring is enabled
        if self.hparams.enable_influence_scoring:
            print(f"This batch originally has {len(batch['dataset_index'])} samples")
            batch = self._filter_low_influence_samples(batch, num_to_remove=self.hparams.num_low_influence_to_remove)
            print(f"This batch after filtering has {len(batch['dataset_index'])} samples")
        else:
            print(f"Influence scoring disabled - using full batch with {len(batch['dataset_index'])} samples")

        output = self(
            **{field: batch[field] for field in list(self.train_seq_fields) + ["sample_id"]}
        )
        loss = self.hparams.loss_func(
            pred=output,
            target=batch["label"],
            observed_mask=batch["label_observed_mask"],
            prediction_mask=batch["prediction_mask"],
            sample_id=batch["sample_id"],
            variate_id=batch["variate_id"],
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss
    
    def on_train_batch_start(self, batch, batch_idx):
        """Calculate per-example gradients for influence function computation and filter low-influence samples"""
        
        # Only compute per-example gradients during training
        if not self.training:
            return
            
        # Skip influence scoring if disabled
        if not self.hparams.enable_influence_scoring:
            print(f"Influence scoring disabled - skipping gradient computation for batch {batch_idx}")
            return
        
        # Get original dataset indices from the new field
        dataset_indices = batch.get("dataset_index", None)
        if dataset_indices is None:
            print(f"Warning: No dataset_index found in batch {batch_idx}")
            print(f"Make sure to use PadCollateWithDatasetIndex and TimeSeriesDatasetWithIndex")
            return
        
        # Get unique dataset indices (filter out padding with -1)
        unique_indices = dataset_indices.flatten().unique()
        unique_indices = unique_indices[unique_indices >= 0]  # Remove padding (-1)
        
        # BOUNDS CHECK: Validate indices are within dataset range
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
            dataset_size = len(self.trainer.datamodule.train_dataset)
            invalid_indices = unique_indices[unique_indices >= dataset_size]
            if len(invalid_indices) > 0:
                print(f"WARNING: Found invalid indices {invalid_indices.tolist()} >= dataset size {dataset_size} in batch {batch_idx}")
                unique_indices = unique_indices[unique_indices < dataset_size]  # Filter out invalid indices
        
        print(f"Unique dataset indices in batch {batch_idx}: {unique_indices.tolist()}")
        print(f"Unique dataset indices in batch {batch_idx} length: {len(unique_indices)}")
        
        # Initialize per-example gradient storage if not exists
        if not hasattr(self, 'per_example_gradients'):
            self.per_example_gradients = {}
            
        # Initialize influence score history if not exists
        if not hasattr(self, 'influence_score_history'):
            self.influence_score_history = {}
            
        # CLEAN UP: Remove any stale indices from previous runs on first batch
        if batch_idx == 0 and hasattr(self, 'per_example_gradients'):
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                dataset_size = len(self.trainer.datamodule.train_dataset)
                stale_indices = [idx for idx in self.per_example_gradients.keys() if idx >= dataset_size]
                for idx in stale_indices:
                    del self.per_example_gradients[idx]
                if stale_indices:
                    print(f"CLEANUP: Cleared {len(stale_indices)} stale influence scores from previous runs")
        
        # Calculate per-sample gradients using original dataset indices
        per_sample_grads = self._compute_per_sample_gradients_with_indices(batch, unique_indices)
        
        # Store gradients indexed by original dataset index
        for i, dataset_idx in enumerate(unique_indices):
            dataset_idx_item = dataset_idx.item()
            
            # Extract gradients for this specific sample
            sample_grads = {}
            for param_name, grad_batch in per_sample_grads.items():
                if grad_batch is not None and i < len(grad_batch):
                    sample_grads[param_name] = grad_batch[i].clone().detach()
            
            # Store in per-example gradient dict
            if dataset_idx_item not in self.per_example_gradients:
                self.per_example_gradients[dataset_idx_item] = []
            
            self.per_example_gradients[dataset_idx_item].append({
                'gradients': sample_grads,
                'step': self.global_step,
                'epoch': self.current_epoch,
                'loss': None  # We don't have outputs yet in on_train_batch_start
            })

        # directly calculate the influence scores on validation gradients
        val_gradients = self.get_validation_gradients_from_trainer()
        influence_scores = self.compute_influence_scores(val_gradients)

        # update the influence scores, but append the new scores to the end, don't overwrite the existing scores
        for sample_idx, scores in influence_scores.items():
            if sample_idx in self.influence_scores:
                self.influence_scores[sample_idx].extend(scores)
            else:
                self.influence_scores[sample_idx] = scores

        # aggregate the influence scores by dataset name
        dataset_influence_scores = {}
        count_dataset_scores = {}
        for sample_idx, scores in self.influence_scores.items():
            for score in scores:
                dataset_name = score.get('dataset_name', 'Unknown')
                if dataset_name not in dataset_influence_scores:
                    dataset_influence_scores[dataset_name] = 0
                    count_dataset_scores[dataset_name] = 0
                # use running average to aggregate the influence scores
                dataset_influence_scores[dataset_name] = (dataset_influence_scores[dataset_name] * count_dataset_scores[dataset_name] + score['influence_score']) / (count_dataset_scores[dataset_name] + 1)
                count_dataset_scores[dataset_name] += 1
        
        # sort the dataset_influence_scores by the score
        dataset_influence_scores = sorted(dataset_influence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # print the dataset_influence_scores in order
        print("=" * 80)
        print("DATASET INFLUENCE SCORES:")
        for dataset_name, score in dataset_influence_scores:
            print(f"  {dataset_name:25} | Score: {score:10.4f} | Step: {self.global_step:6d} | Epoch: {self.current_epoch:6d} | Count: {count_dataset_scores[dataset_name]:6d}")
        print("=" * 80)

        # save dataset_influence_scores to a csv file with header
        with open('dataset_influence_scores_20250619.csv', 'a') as f:
            f.write("dataset_name,score,step,epoch,count\n")
            for dataset_name, score in dataset_influence_scores:
                f.write(f"{dataset_name},{score},{self.global_step},{self.current_epoch},{count_dataset_scores[dataset_name]}\n")
            
        # Update influence score history for future filtering
        self._update_influence_score_history(influence_scores)
        
        # clear the per-example gradients
        self.clear_per_example_gradients()

    def _filter_low_influence_samples(self, batch, num_to_remove=16):
        """Filter out samples with lowest influence scores from the current batch"""
        
        # Skip filtering if influence scoring is disabled
        if not self.hparams.enable_influence_scoring:
            print("Influence scoring disabled - skipping batch filtering")
            return batch
            
        if "dataset_index" not in batch:
            print("Warning: No dataset_index found in batch, skipping filtering")
            return batch
        
        dataset_indices = batch["dataset_index"]
        batch_size, seq_len = dataset_indices.shape
        
        # Get unique dataset indices in this batch
        unique_indices = dataset_indices.flatten().unique()
        unique_indices = unique_indices[unique_indices >= 0]  # Remove padding (-1)
        
        if len(unique_indices) <= num_to_remove:
            print(f"Batch has only {len(unique_indices)} unique samples, not removing any")
            return batch
        
        # Get influence scores for samples in this batch
        sample_scores = []
        for idx in unique_indices:
            idx_item = idx.item()
            
            # Get latest influence score for this sample
            if (hasattr(self, 'influence_score_history') and 
                idx_item in self.influence_score_history):
                latest_score = self.influence_score_history[idx_item]
                sample_scores.append((idx_item, latest_score))
            else:
                # If no history, assign neutral score (0.0)
                sample_scores.append((idx_item, 0.0))
                print(f"No influence score history found for sample {idx_item}")
        
        # Sort by influence score (ascending) and get the lowest scoring samples
        sample_scores.sort(key=lambda x: x[1])
        samples_to_remove = [idx for idx, score in sample_scores[:num_to_remove]]
        
        if len(samples_to_remove) == 0:
            return batch
        
        print(f"Target: removing {num_to_remove} batch positions from low-influence samples: {samples_to_remove}")
        print(f"Average influence score: {sum(score for _, score in sample_scores) / len(sample_scores)}")
        print(f"Samples to remove average influence score: {sum(score for _, score in sample_scores[:num_to_remove]) / num_to_remove}")
        
        # Create mask for samples to keep
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=dataset_indices.device)
        
        # Track how many batch positions we've removed
        removed_count = 0
        
        for sample_idx in samples_to_remove:
            if removed_count >= num_to_remove:
                break
                
            # Find batch positions that contain this sample
            sample_mask = (dataset_indices == sample_idx).any(dim=1)
            sample_positions = sample_mask.nonzero().squeeze(1)
            
            # Remove only one instance of this sample (or fewer if we're at the limit)
            positions_to_remove = min(len(sample_positions), num_to_remove - removed_count)
            if positions_to_remove > 0:
                keep_mask[sample_positions[:positions_to_remove]] = False
                removed_count += positions_to_remove
        
        # Filter all tensors in the batch
        filtered_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.dim() > 0 and value.shape[0] == batch_size:
                    # This tensor has batch dimension, filter it
                    filtered_batch[key] = value[keep_mask]
                else:
                    # This tensor doesn't have batch dimension, keep as is
                    filtered_batch[key] = value
            else:
                # Non-tensor values, keep as is
                filtered_batch[key] = value
        
        original_batch_size = batch_size
        new_batch_size = keep_mask.sum().item()
        actual_removed = original_batch_size - new_batch_size
        
        print(f"Filtered batch size: {original_batch_size} -> {new_batch_size} (actually removed {actual_removed} positions)")
        
        return filtered_batch
    
    def _update_influence_score_history(self, influence_scores):
        """Update the influence score history for batch filtering"""
        
        if not hasattr(self, 'influence_score_history'):
            self.influence_score_history = {}
        
        # Process each sample's influence scores
        for sample_idx, score_entries in influence_scores.items():
            if len(score_entries) > 0:
                # Use the most recent influence score for this sample
                latest_score_entry = score_entries[-1]
                influence_score = latest_score_entry['influence_score']
                
                # Store the latest influence score
                self.influence_score_history[sample_idx] = influence_score
        
        print(f"Updated influence score history for {len(influence_scores)} samples")

    def _compute_per_sample_gradients_with_indices(self, batch, unique_indices):
        """Compute per-sample gradients for samples with specific dataset indices."""
        per_sample_grads = {}
        
        # Initialize gradient storage
        for name, param in self.named_parameters():
            if param.requires_grad:
                per_sample_grads[name] = []
        
        # Get dataset indices tensor
        dataset_indices = batch["dataset_index"]
        
        # Compute gradient for each unique dataset index
        for dataset_idx in unique_indices:
            # Zero gradients
            self.zero_grad()
            
            # Create mask for this dataset index
            mask = (dataset_indices == dataset_idx)
            
            # Find positions where this dataset index appears
            batch_indices, seq_indices = torch.where(mask)
            
            if len(batch_indices) == 0:
                # No data for this index, store None
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        per_sample_grads[name].append(None)
                continue
            
            # Get unique batch indices (samples in the batch containing this dataset index)
            unique_batch_indices = batch_indices.unique()
            
            # Create a mini-batch with only the relevant samples
            single_batch = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    if value.dim() > 1:
                        # Take only the samples that contain this dataset index
                        single_batch[key] = value[unique_batch_indices]
                    else:
                        single_batch[key] = value
                else:
                    single_batch[key] = value
            
            # Forward pass for this dataset index
            try:
                output = self(**{
                    field: single_batch[field] 
                    for field in list(self.train_seq_fields) + ["sample_id"]
                    if field in single_batch
                })
                
                # Compute loss for this dataset index
                loss = self.hparams.loss_func(
                    pred=output,
                    target=single_batch.get("label"),
                    observed_mask=single_batch.get("label_observed_mask"),
                    prediction_mask=single_batch.get("prediction_mask"),
                    sample_id=single_batch.get("sample_id"),
                    variate_id=single_batch.get("variate_id"),
                )
                
                # Scale loss by the proportion of data from this dataset index
                total_elements = mask.sum().item()
                loss = loss * total_elements / mask.numel()
                
                # Backward pass
                loss.backward(retain_graph=True)
                
                # Store gradients
                for name, param in self.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        per_sample_grads[name].append(param.grad.clone().detach())
                    else:
                        per_sample_grads[name].append(None)
                    
            except Exception as e:
                print(f"Error computing gradient for dataset index {dataset_idx}: {e}")
                # Store None for this sample
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        per_sample_grads[name].append(None)
        
        # Convert lists to tensors
        for name in per_sample_grads:
            valid_grads = [g for g in per_sample_grads[name] if g is not None]
            if valid_grads:
                per_sample_grads[name] = torch.stack(valid_grads, dim=0)
            else:
                per_sample_grads[name] = None
        
        # Clear gradients
        self.zero_grad()
        
        return per_sample_grads

    def compute_influence_scores(self, val_gradients):
        """Compute influence scores by inner product with validation gradients"""
        if not hasattr(self, 'per_example_gradients'):
            print("No per-example gradients stored")
            return {}
        
        influence_scores = {}
        
        # Try to get dataset metadata for mapping global indices to dataset names
        dataset_metadata = None
        try:
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                train_dataset = self.trainer.datamodule.train_dataset
                # Navigate to the ConcatDatasetBuilderWithGlobalIndex if it exists
                # This assumes the train_dataset was created by instantiating a config with a ConcatDatasetBuilderWithGlobalIndex
                # We need to trace back to find the dataset builder that created this dataset
                if hasattr(train_dataset, 'datasets'):  # ConcatDataset
                    # Look for global index metadata in any of the sub-datasets
                    for sub_dataset in train_dataset.datasets:
                        if hasattr(sub_dataset, 'global_offset'):
                            # This indicates we're using the enhanced datasets with global indexing
                            # We need to find the builder that created the dataset hierarchy
                            pass
        except Exception as e:
            print(f"Warning: Could not access dataset metadata for sub-dataset names: {e}")
        
        for sample_idx, grad_history in self.per_example_gradients.items():
            sample_scores = []
            
            for grad_entry in grad_history:
                train_grads = grad_entry['gradients']
                
                # Compute inner product with validation gradients
                inner_product = 0.0
                param_count = 0
                
                for param_name in train_grads:
                    if param_name in val_gradients and train_grads[param_name] is not None:
                        train_grad = train_grads[param_name].flatten()
                        val_grad = val_gradients[param_name].flatten()
                        
                        # Ensure same size
                        if train_grad.shape == val_grad.shape:
                            inner_product += torch.dot(train_grad, val_grad).item()
                            param_count += 1
                
                if param_count > 0:
                    score_entry = {
                        'influence_score': inner_product,
                        'step': grad_entry['step'],
                        'epoch': grad_entry['epoch'],
                        'loss': None,  # We don't have outputs yet in on_train_batch_start
                        'global_dataset_index': sample_idx  # Store the global index
                    }
                    
                    # Try to add dataset name if we can map global index to dataset
                    if dataset_metadata and sample_idx in dataset_metadata:
                        score_entry['dataset_name'] = dataset_metadata[sample_idx]
                    else:
                        # Fallback: try to map using a simpler approach
                        score_entry['dataset_name'] = self._get_dataset_name_for_index(sample_idx)
                    
                    sample_scores.append(score_entry)
            
            influence_scores[sample_idx] = sample_scores
        
        return influence_scores
    
    def _get_dataset_name_for_index(self, global_idx: int) -> str:
        """Helper method to get dataset name for a global index."""
        try:
            # BOUNDS CHECK: Validate global_idx is within reasonable range
            max_dataset_size = 20000  # Reasonable upper bound
            if global_idx > max_dataset_size:
                print(f"WARNING: Global index {global_idx} exceeds reasonable bounds (>{max_dataset_size}). This might indicate stale influence scores.")
                return f"OutOfBounds_Idx_{global_idx}"
            
            # Method 1: Try to use ConcatDatasetBuilderWithGlobalIndex metadata (preferred)
            if (hasattr(self.trainer, 'datamodule') and 
                hasattr(self.trainer.datamodule, 'data_builder') and
                hasattr(self.trainer.datamodule.data_builder, 'get_dataset_name_for_global_index')):
                
                # Additional bounds check using actual dataset size
                if hasattr(self.trainer.datamodule, 'train_dataset'):
                    dataset_size = len(self.trainer.datamodule.train_dataset)
                    if global_idx >= dataset_size:
                        print(f"WARNING: Global index {global_idx} >= dataset size {dataset_size}. Clearing stale influence scores.")
                        # Clear stale per_example_gradients to prevent future issues
                        if hasattr(self, 'per_example_gradients'):
                            # Remove any indices beyond the current dataset size
                            stale_indices = [idx for idx in self.per_example_gradients.keys() if idx >= dataset_size]
                            for idx in stale_indices:
                                del self.per_example_gradients[idx]
                            if stale_indices:
                                print(f"Cleared {len(stale_indices)} stale influence scores with indices: {stale_indices[:10]}{'...' if len(stale_indices) > 10 else ''}")
                        return f"Stale_Idx_{global_idx}_Cleared"
                
                return self.trainer.datamodule.data_builder.get_dataset_name_for_global_index(global_idx)
            
            # Method 2: Fallback to manual traversal of ConcatDataset
            elif hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                train_dataset = self.trainer.datamodule.train_dataset
                
                # Bounds check against actual dataset
                if global_idx >= len(train_dataset):
                    print(f"WARNING: Global index {global_idx} >= actual dataset size {len(train_dataset)}")
                    return f"OutOfRange_Idx_{global_idx}"
                
                # Check if it's a ConcatDataset with sub-datasets that have global_offset
                if hasattr(train_dataset, 'datasets'):
                    cumulative_size = 0
                    for i, sub_dataset in enumerate(train_dataset.datasets):
                        dataset_size = len(sub_dataset)
                        if global_idx < cumulative_size + dataset_size:
                            # This global index belongs to this sub-dataset
                            # Try to get the dataset name from various sources
                            
                            # Method 2a: Check if dataset has indexer with dataset info
                            if hasattr(sub_dataset, 'indexer') and hasattr(sub_dataset.indexer, 'dataset'):
                                if hasattr(sub_dataset.indexer.dataset, 'info') and hasattr(sub_dataset.indexer.dataset.info, 'dataset_name'):
                                    return sub_dataset.indexer.dataset.info.dataset_name
                            
                            # Method 2b: Check if dataset itself has info
                            if hasattr(sub_dataset, 'info') and hasattr(sub_dataset.info, 'dataset_name'):
                                return sub_dataset.info.dataset_name
                            
                            # Method 2c: Return a descriptive name based on position
                            return f"SubDataset_{i}"
                        
                        cumulative_size += dataset_size
            
            # Final fallback
            return f"Dataset_GlobalIdx_{global_idx}"
            
        except Exception as e:
            return f"Unknown_Idx_{global_idx}_Error_{str(e)[:30]}"

    def get_validation_gradients_from_trainer(self):
        """Compute gradients using trainer's validation dataloader"""
        
        if not hasattr(self.trainer, 'val_dataloaders') or not self.trainer.val_dataloaders:
            print("Warning: No validation dataloader found in trainer")
            return {}
        
        # Get validation dataloader from trainer
        val_dataloader = self.trainer.val_dataloaders[0]  # Use first validation dataloader
        
        return self.get_validation_gradients(val_dataloader)

    def get_validation_gradients(self, dataloader_or_dataset):
        """Compute gradients on entire validation dataset for influence computation"""
        
        # Save current training state
        was_training = self.training
        self.eval()
        self.zero_grad()
        
        # Handle both dataset and dataloader inputs
        if hasattr(dataloader_or_dataset, '__iter__') and hasattr(dataloader_or_dataset, '__len__'):
            # It's a dataloader
            val_dataloader = dataloader_or_dataset
        else:
            # It's a dataset, create dataloader
            from torch.utils.data import DataLoader
            val_dataloader = DataLoader(
                dataloader_or_dataset,
                batch_size=32,  # Smaller batch size for memory efficiency
                shuffle=False,
                collate_fn=getattr(dataloader_or_dataset, 'collate_fn', None)
            )
        
        total_samples = 0
        accumulated_gradients = {}
        
        print(f"Computing validation gradients over {len(val_dataloader)} batches...")
        print(f"NOTE: Validation dataset indices will NOT interfere with training influence scores")
        
        # SOLUTION: Manual gradient accumulation with immediate cleanup
        for batch_idx, val_batch in enumerate(val_dataloader):
            try:
                # Clear gradients for this batch
                self.zero_grad()
                
                # IMPORTANT: Remove dataset_index from validation batch to prevent contamination
                # Validation dataset indices should NOT be used for influence computation
                val_batch_clean = {k: v for k, v in val_batch.items() if k != 'dataset_index'}
                # if 'dataset_index' in val_batch:
                #     # Log warning about validation dataset index contamination
                #     val_indices = val_batch['dataset_index'].flatten().unique()
                #     val_indices_valid = val_indices[val_indices >= 0]
                #     if len(val_indices_valid) > 0 and batch_idx == 0:  # Only log once
                #         max_val_idx = val_indices_valid.max().item()
                #         print(f"INFO: Validation batch contains dataset indices up to {max_val_idx}")
                #         print(f"INFO: These indices are EXCLUDED from influence computation to prevent contamination")
                
                # Move tensors to device (consider CPU for very large models)
                val_batch_clean = {
                    k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in val_batch_clean.items()
                }
                
                # Forward pass with gradient computation
                with torch.enable_grad():
                    output = self(**{
                        field: val_batch_clean[field] 
                        for field in list(self.train_seq_fields) + ["sample_id"]
                        if field in val_batch_clean
                    })
                    
                    # Compute loss for this batch
                    batch_loss = self.hparams.loss_func(
                        pred=output,
                        target=val_batch_clean["label"],
                        observed_mask=val_batch_clean["label_observed_mask"],
                        prediction_mask=val_batch_clean["prediction_mask"],
                        sample_id=val_batch_clean["sample_id"],
                        variate_id=val_batch_clean["variate_id"],
                    )
                    
                    # Get actual batch size for proper weighting
                    batch_size = (
                        val_batch_clean["sample_id"].max(dim=1).values.sum().item() 
                        if "sample_id" in val_batch_clean and val_batch_clean["sample_id"].dim() > 1
                        else val_batch_clean["target"].shape[0]
                    )
                    
                    # Backward pass - this computes gradients for this batch only
                    batch_loss.backward()
                
                # Manually accumulate gradients
                for name, param in self.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad = param.grad.clone().detach() * batch_size  # Weight by batch size
                        
                        if name not in accumulated_gradients:
                            accumulated_gradients[name] = grad
                        else:
                            accumulated_gradients[name] += grad
                
                total_samples += batch_size
                
                # Clear intermediate variables to free memory
                del output, batch_loss, val_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(val_dataloader)} validation batches")
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        if total_samples == 0:
            print("Warning: No validation samples processed")
            return {}
        
        # Average the accumulated gradients
        val_gradients = {}
        for name, accumulated_grad in accumulated_gradients.items():
            val_gradients[name] = accumulated_grad / total_samples
        
        # Clear gradients and restore training state
        self.zero_grad()
        if was_training:
            self.train()
        
        print(f"Computed validation gradients for {len(val_gradients)} parameters over {total_samples} samples")
        
        return val_gradients

    # Optional: Add memory management
    # def clear_per_example_gradients(self, keep_recent_steps=1):
    #     """Clear old per-example gradients to manage memory"""
    #     if not hasattr(self, 'per_example_gradients'):
    #         return
        
    #     current_step = self.global_step
        
    #     for sample_idx in list(self.per_example_gradients.keys()):
    #         # Keep only recent gradients
    #         recent_grads = [
    #             grad_entry for grad_entry in self.per_example_gradients[sample_idx]
    #             if current_step - grad_entry['step'] <= keep_recent_steps
    #         ]
            
    #         if recent_grads:
    #             self.per_example_gradients[sample_idx] = recent_grads
    #         else:
    #             del self.per_example_gradients[sample_idx]
        
    #     print(f"Cleared old gradients, keeping {keep_recent_steps} recent steps")

    def clear_per_example_gradients(self, keep_recent_steps=None):
        """Clear old per-example gradients to manage memory"""
        if not hasattr(self, 'per_example_gradients'):
            return
        
        self.per_example_gradients = {}

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        distr, preds = self.infer(
            **{field: batch[field] for field in list(self.train_seq_fields) + ["sample_id"]}
        )
        val_loss = self.hparams.loss_func(
            pred=distr,
            target=batch["label"],
            observed_mask=batch["label_observed_mask"],
            **{
                field: batch[field]
                for field in [
                    "prediction_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
            add_dataloader_idx=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                metric = metric_func(
                    pred=preds,
                    target=batch["label"],
                    observed_mask=batch["label_observed_mask"],
                    **{
                        field: batch[field]
                        for field in [
                            "prediction_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                    add_dataloader_idx=True,
                )

        return val_loss
    
    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            nn.Linear,
        )
        blacklist_params = (
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }
        
    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        def default_train_transform():
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=(),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_size=self.module.patch_size,
                    patch_size_constraints=None,
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=(),
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=(),
                    optional_truncate_fields=(),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackCollection(
                    field="label_observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="label",
                    fields=("label",),
                    optional_fields=(),
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields) + ["_dataset_idx"])
            )

        return defaultdict(lambda: default_train_transform)
    
    @property
    def val_transform_map(
        self,
    ) -> dict[str | type, Callable[..., Transformation]]:
        def default_val_transform(
            offset: int,
            distance: int,
            prediction_length: int,
            context_length: int,
            patch_size: int,
        ):
            return (
                SampleDimension(
                    max_dim=1,
                    fields=("target",),
                    optional_fields=(),
                )
                + GetPatchSize(
                    min_time_patches=2,
                    target_field="target",
                    patch_size=self.module.patch_size,
                    patch_size_constraints=None,
                    offset=True,
                )
                + EvalCrop_AdaLength(
                    offset,
                    distance,
                    prediction_length,
                    context_length,
                    fields=("target",),
                    optional_fields=(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                )
                + EvalPad_AdaLength(
                    prediction_length=prediction_length,
                    context_length=context_length,
                    patch_size=self.module.patch_size,
                    fields=("target",),
                    optional_fields=()
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=(),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + EvalMaskedPrediction(
                    mask_length=math.ceil(prediction_length / patch_size),
                    target_field="target",
                    truncate_fields=(),
                    optional_truncate_fields=(),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=(),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=(),
                    feat=True,
                )
                + FlatPackCollection(
                    field="label_observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="label",
                    fields=("label",),
                    optional_fields=(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields) + ["_dataset_idx"])
            )

        return defaultdict(lambda: default_val_transform)