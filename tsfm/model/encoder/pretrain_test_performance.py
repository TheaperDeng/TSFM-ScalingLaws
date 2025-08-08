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
import time
from tsfm.data.augmentation import mixup
import numpy as np

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
        enable_influence_scoring: bool = False,
        enable_dataset_contribution_logging: bool = False,
        enable_reweighting: bool = False,
        influence_filter_frequency: int = 1,  # temporary
        use_cosine_similarity: bool = False,  # New parameter for cosine similarity
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
        self.recommended_weights = {}  # Initialize for recommended weights-based filtering
        self.threshold = 4000
        self.cache_val_batch = None  # This is used to cache the validation batch for TS influence scoring
        
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
        # Determine whether to use influence-based or random filtering
        current_step = self.global_step
        use_influence_filtering = (current_step % self.hparams.influence_filter_frequency == 0) and self.hparams.enable_influence_scoring
        
        # Always apply some form of filtering (influence-based every N steps, random otherwise)
        print(f"Step {current_step}: This batch originally has {len(batch['dataset_index'])} samples")
        
        if use_influence_filtering:
            print(f"Step {current_step}: Using influence-based filtering (every {self.hparams.influence_filter_frequency} steps)")
            batch = self._filter_low_influence_samples(
                batch, 
                num_to_remove=self.hparams.num_low_influence_to_remove, 
                use_influence_scores=True
            )
        else:
            # Determine filtering strategy based on available recommended weights
            if hasattr(self, 'recommended_weights') and self.recommended_weights:
                print(f"Step {current_step}: Using recommended weights-based filtering")
            else:
                print(f"Step {current_step}: Using random filtering (no recommended weights available yet)")
            
            batch = self._filter_low_influence_samples(
                batch, 
                num_to_remove=self.hparams.num_low_influence_to_remove, 
                use_influence_scores=False
            )
        
        print(f"Step {current_step}: This batch after filtering has {len(batch['dataset_index'])} samples")

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

        # update influence_filter_frequency according to global_step
        # increase the influence_filter_frequency by 1 every 1000 steps, then 2000, then 3000, ...
        # self.hparams.influence_filter_frequency = int(self.global_step / self.threshold) + 1
        # self.threshold = self.hparams.influence_filter_frequency * 1000
        
        # if self.global_step % self.threshold == self.threshold - 1:
        #     self.hparams.influence_filter_frequency += 1
        #     self.threshold = 1000 * (self.hparams.influence_filter_frequency) + self.threshold
        # print(f"Step {self.global_step}: Threshold updated to {self.threshold}")
        # print(f"Step {self.global_step}: Influence filter frequency updated to {self.hparams.influence_filter_frequency}")

        # Only compute per-example gradients during training
        if not self.training:
            return
        
        # Only compute influence scores when influence-based filtering will be used
        current_step = self.global_step
        use_influence_filtering = (current_step % self.hparams.influence_filter_frequency == 0)
        
        if not use_influence_filtering:
            print(f"Step {current_step}: Skipping influence computation (random filtering step)")
            return
            
        # Skip influence scoring if disabled
        if not self.hparams.enable_influence_scoring:
            print(f"Step {current_step}: Influence scoring disabled - skipping gradient computation")
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
        influence_scores = self.compute_influence_scores_batched(val_gradients)

        # update the influence scores, but append the new scores to the end, don't overwrite the existing scores
        for sample_idx, scores in influence_scores.items():
            if sample_idx in self.influence_scores:
                self.influence_scores[sample_idx].extend(scores)
            else:
                self.influence_scores[sample_idx] = scores

        # Clean up old influence scores to keep only recent 4000 steps
        self._cleanup_old_influence_scores(keep_recent_steps=4000)

        # Only compute and log dataset contributions if dataset contribution logging is enabled
        if self.hparams.enable_dataset_contribution_logging:
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

            # # save dataset_influence_scores to a csv file with header
            # with open('dataset_influence_scores_20250619.csv', 'a') as f:
            #     f.write("dataset_name,score,step,epoch,count\n")
            #     for dataset_name, score in dataset_influence_scores:
            #         f.write(f"{dataset_name},{score},{self.global_step},{self.current_epoch},{count_dataset_scores[dataset_name]}\n")
        else:
            print("Dataset contribution logging disabled (enable_dataset_contribution_logging=False)")

        if self.hparams.enable_reweighting:
            # Calculate updated sampling ratios based on dataset influence scores

            scores = np.array([score for _, score in dataset_influence_scores])
            dataset_names = [name for name, _ in dataset_influence_scores]
            
            # Method 1: Linear scaling (more conservative)
            # Normalize to [0.1, 2.0] range to avoid extreme ratios
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
                linear_ratios = 0.1 + 1.9 * normalized_scores  # Scale to [0.1, 2.0]
                linear_ratios = linear_ratios / linear_ratios.mean()  # Normalize so mean = 1.0
            else:
                linear_ratios = np.ones(len(scores))  # All equal if no variation
            
            # Create sampling ratio dictionary
            sampling_ratios = {}
            for i, dataset_name in enumerate(dataset_names):
                sampling_ratios[dataset_name] = {
                    'influence_score': scores[i],
                    'linear_ratio': float(linear_ratios[i]),
                    'count': count_dataset_scores[dataset_name]
                }
            
            # Store sampling ratios for potential use by external components
            self.latest_sampling_ratios = sampling_ratios
            
            # Option: Use linear ratios as the recommended sampling weights
            recommended_weights = {name: info['linear_ratio'] for name, info in sampling_ratios.items()}
            print(f"Recommended sampling weights: {recommended_weights}")
            
            # Save recommended weights as member variable for future filtering rounds
            self.recommended_weights = recommended_weights
            
            # Dataset weights calculation completed - weights not applied to maintain decoupling
        else:
            print("Dataset reweighting disabled (enable_reweighting=False)")

        # Update influence score history for future filtering
        self._update_influence_score_history(influence_scores)

        # # clear the per-example gradients
        # self.clear_per_example_gradients()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Clear per-example gradients after training step"""
        self.clear_per_example_gradients()

    def _filter_low_influence_samples(self, batch, num_to_remove=16, use_influence_scores=True):
        """Filter out samples from the current batch using influence scores or random selection"""
        
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
        
        if use_influence_scores:
            # Use influence-based filtering (original behavior)
            if not self.hparams.enable_influence_scoring:
                print("Warning: Influence scoring disabled but influence-based filtering requested, using random filtering instead")
                use_influence_scores = False
            else:
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
                
                print(f"Influence-based filtering: removing {num_to_remove} batch positions from low-influence samples: {samples_to_remove}")
                print(f"Average influence score: {sum(score for _, score in sample_scores) / len(sample_scores)}")
                print(f"Samples to remove average influence score: {sum(score for _, score in sample_scores[:num_to_remove]) / num_to_remove}")
        
        if not use_influence_scores:
            # Use recommended weights-based filtering if available, otherwise random filtering
            if hasattr(self, 'recommended_weights') and self.recommended_weights:
                # Get dataset names and weights for samples in this batch
                sample_weights = []
                for idx in unique_indices:
                    idx_item = idx.item()
                    dataset_name = self._get_dataset_name_for_index(idx_item)
                    weight = self.recommended_weights.get(dataset_name, 1.0)  # Default to 1.0 if not found
                    sample_weights.append((idx_item, weight, dataset_name))
                
                # Use probabilistic sampling based on inverse weights to maintain diversity
                # Lower weights = higher probability of removal, but still maintains randomness
                import numpy as np
                
                # Extract weights and indices
                indices = [idx for idx, weight, dataset_name in sample_weights]
                weights = np.array([weight for idx, weight, dataset_name in sample_weights])
                dataset_names = [dataset_name for idx, weight, dataset_name in sample_weights]
                
                # Calculate removal probabilities (inverse of weights, normalized)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-6
                inverse_weights = 1.0 / (weights + epsilon)
                removal_probs = inverse_weights / inverse_weights.sum()
                
                # Sample indices to remove based on probabilities
                try:
                    selected_indices = np.random.choice(
                        len(indices), 
                        size=min(num_to_remove, len(indices)), 
                        replace=False, 
                        p=removal_probs
                    )
                    samples_to_remove = [indices[i] for i in selected_indices]
                    
                    print(f"Recommended weights-based filtering: probabilistically removing {len(samples_to_remove)} batch positions")
                    for i in selected_indices:
                        idx, weight, dataset_name = indices[i], weights[i], dataset_names[i]
                        prob = removal_probs[i]
                        print(f"  Removing sample {idx} from {dataset_name} (weight: {weight:.3f}, removal_prob: {prob:.3f})")
                        
                except Exception as e:
                    print(f"Error in probabilistic sampling: {e}, falling back to random selection")
                    import random
                    samples_to_remove = random.sample(indices, min(num_to_remove, len(indices)))
            else:
                # Fallback to random filtering if no recommended weights available
                import random
                unique_indices_list = unique_indices.cpu().numpy().tolist()
                samples_to_remove = random.sample(unique_indices_list, min(num_to_remove, len(unique_indices_list)))
                
                print(f"Random filtering (no recommended weights available): removing {len(samples_to_remove)} batch positions from randomly selected samples: {samples_to_remove}")
        
        if len(samples_to_remove) == 0:
            return batch

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
        
        # Determine filter type for logging
        if use_influence_scores:
            filter_type = "influence-based"
        elif hasattr(self, 'recommended_weights') and self.recommended_weights:
            filter_type = "recommended weights-based"
        else:
            filter_type = "random"
        
        print(f"Filtered batch size ({filter_type}): {original_batch_size} -> {new_batch_size} (actually removed {actual_removed} positions)")
        
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

    def _cleanup_old_influence_scores(self, keep_recent_steps=1600):
        """Clean up old influence scores to keep only recent steps for memory management"""
        
        if not hasattr(self, 'influence_scores') or not self.influence_scores:
            return
        
        current_step = self.global_step
        cutoff_step = current_step - keep_recent_steps
        
        # Count entries before cleanup
        total_entries_before = sum(len(scores) for scores in self.influence_scores.values())
        samples_before = len(self.influence_scores)
        
        # Clean up old entries from each sample's influence scores
        samples_to_remove = []
        
        for sample_idx, score_entries in self.influence_scores.items():
            # Filter out old entries based on step
            recent_entries = [
                entry for entry in score_entries 
                if entry.get('step', 0) > cutoff_step
            ]
            
            if recent_entries:
                # Keep only recent entries
                self.influence_scores[sample_idx] = recent_entries
            else:
                # Mark sample for removal if no recent entries
                samples_to_remove.append(sample_idx)
        
        # Remove samples with no recent entries
        for sample_idx in samples_to_remove:
            del self.influence_scores[sample_idx]
        
        # Count entries after cleanup
        total_entries_after = sum(len(scores) for scores in self.influence_scores.values())
        samples_after = len(self.influence_scores)
        
        entries_removed = total_entries_before - total_entries_after
        samples_removed = samples_before - samples_after
        
        if entries_removed > 0 or samples_removed > 0:
            print(f"Cleaned up influence scores: removed {entries_removed} old entries from {samples_removed} samples")
            print(f"Kept influence scores from step {cutoff_step + 1} onwards (recent {keep_recent_steps} steps)")
            print(f"Remaining: {total_entries_after} entries across {samples_after} samples")

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
                print(f"WARNING: No data for dataset index {dataset_idx}, skipping...")
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
                
                # Compute inner product or cosine similarity with validation gradients
                similarity_score = 0.0
                param_count = 0
                
                if self.hparams.use_cosine_similarity:
                    # Compute cosine similarity for each parameter separately and average
                    for param_name in train_grads:
                        if param_name in val_gradients and train_grads[param_name] is not None:
                            train_grad = train_grads[param_name].flatten()
                            val_grad = val_gradients[param_name].flatten()
                            
                            # Ensure same size
                            if train_grad.shape == val_grad.shape:
                                # Compute cosine similarity: cos(θ) = (A·B) / (||A|| * ||B||)
                                train_norm = torch.norm(train_grad)
                                val_norm = torch.norm(val_grad)
                                
                                if train_norm > 0 and val_norm > 0:
                                    cosine_sim = torch.dot(train_grad, val_grad) / (train_norm * val_norm)
                                    similarity_score += cosine_sim.item()
                                    param_count += 1
                else:
                    # Original dot product computation
                    for param_name in train_grads:
                        if param_name in val_gradients and train_grads[param_name] is not None:
                            train_grad = train_grads[param_name].flatten()
                            val_grad = val_gradients[param_name].flatten()
                            
                            # Ensure same size
                            if train_grad.shape == val_grad.shape:
                                similarity_score += torch.dot(train_grad, val_grad).item()
                                param_count += 1
                
                if param_count > 0:
                    score_entry = {
                        'influence_score': similarity_score,
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
    
    def compute_influence_scores_batched(self, val_gradients):
        """Compute influence scores using batched operations for major speedup"""
        if not hasattr(self, 'per_example_gradients'):
            print("No per-example gradients stored")
            return {}
        
        if not val_gradients:
            print("No validation gradients provided")
            return {}
        
        # Pre-process validation gradients - get consistent parameter order
        param_names = sorted([n for n in val_gradients.keys() if val_gradients[n] is not None])
        if not param_names:
            print("No valid validation gradients found")
            return {}
        
        val_grad_flat = torch.cat([val_gradients[name].flatten() for name in param_names])
        device = val_grad_flat.device
        
        # Collect all training gradients into batches
        all_sample_indices = []
        all_train_grads = []
        all_metadata = []
        
        for sample_idx, grad_history in self.per_example_gradients.items():
            for grad_entry in grad_history:
                train_grads = grad_entry['gradients']
                
                # Check if all required gradients exist and are valid
                valid_grads = []
                valid = True
                
                for name in param_names:
                    if name in train_grads and train_grads[name] is not None:
                        valid_grads.append(train_grads[name].flatten())
                    else:
                        valid = False
                        break
                
                if valid and len(valid_grads) == len(param_names):
                    try:
                        train_grad_flat = torch.cat(valid_grads)
                        # Ensure same device
                        train_grad_flat = train_grad_flat.to(device)
                        
                        all_sample_indices.append(sample_idx)
                        all_train_grads.append(train_grad_flat)
                        all_metadata.append(grad_entry)
                    except Exception as e:
                        print(f"Warning: Could not process gradients for sample {sample_idx}: {e}")
                        continue
        
        if not all_train_grads:
            print("No valid training gradients found for batch processing")
            return {}
        
        print(f"Batch processing influence scores for {len(all_train_grads)} gradient entries...")
        
        try:
            # Stack into matrix: [num_samples, num_params]
            train_grad_matrix = torch.stack(all_train_grads)
            
            if self.hparams.use_cosine_similarity:
                # Normalize training gradients (each row)
                train_grad_norms = torch.norm(train_grad_matrix, dim=1, keepdim=True)
                # Avoid division by zero
                train_grad_norms = torch.clamp(train_grad_norms, min=1e-8)
                train_grad_matrix_normalized = train_grad_matrix / train_grad_norms
                
                # Normalize validation gradients
                val_grad_norm = torch.norm(val_grad_flat)
                val_grad_norm = torch.clamp(val_grad_norm, min=1e-8)
                val_grad_flat_normalized = val_grad_flat / val_grad_norm
                
                # Compute cosine similarity: [num_samples]
                influence_scores_flat = torch.matmul(train_grad_matrix_normalized, val_grad_flat_normalized)
            else:
                # Original dot product computation: [num_samples]
                influence_scores_flat = torch.matmul(train_grad_matrix, val_grad_flat)
            
            # Process results back into the expected format
            influence_scores = {}
            for i, (sample_idx, metadata) in enumerate(zip(all_sample_indices, all_metadata)):
                score_entry = {
                    'influence_score': influence_scores_flat[i].item(),
                    'step': metadata['step'],
                    'epoch': metadata['epoch'],
                    'loss': metadata.get('loss', None),
                    'global_dataset_index': sample_idx
                }
                
                # Try to add dataset name
                try:
                    score_entry['dataset_name'] = self._get_dataset_name_for_index(sample_idx)
                except Exception as e:
                    score_entry['dataset_name'] = f"Unknown_Idx_{sample_idx}"
                
                if sample_idx not in influence_scores:
                    influence_scores[sample_idx] = []
                influence_scores[sample_idx].append(score_entry)
            
            print(f"Successfully computed influence scores for {len(influence_scores)} unique samples")
            return influence_scores
            
        except Exception as e:
            print(f"Error in batched influence computation: {e}")
            print("Falling back to original method...")
            return self.compute_influence_scores(val_gradients)
    
    def _get_dataset_name_for_index(self, global_idx: int) -> str:
        """Helper method to get dataset name for a global index."""
        try:
            # BOUNDS CHECK: Validate global_idx is within reasonable range
            dataset_size = len(self.trainer.datamodule.train_dataset)  # Reasonable upper bound
            if global_idx > dataset_size:
                print(f"WARNING: Global index {global_idx} exceeds reasonable bounds (>{dataset_size}). This might indicate stale influence scores.")
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

    def get_validation_gradients_from_trainer(self, max_val_samples=32):  # None
        """Compute gradients using trainer's validation dataloader
        
        Args:
            max_val_samples: Maximum number of validation samples to use. If None, uses all samples.
        """
        
        if not hasattr(self.trainer, 'val_dataloaders') or not self.trainer.val_dataloaders:
            print("Warning: No validation dataloader found in trainer")
            return {}
        
        # Get validation dataloader from trainer
        val_dataloader = self.trainer.val_dataloaders[0]  # Use first validation dataloader
        
        # Check sampler for shuffle as well
        if hasattr(val_dataloader, 'sampler') and hasattr(val_dataloader.sampler, 'shuffle'):
            if val_dataloader.sampler.shuffle:
                print("WARNING: Validation dataloader sampler has shuffle=True!")
                print("This may still cause non-deterministic ordering even if shuffle=false in config.")
            print("Val dataloader sampler shuffle: ", val_dataloader.sampler.shuffle)

        return self.get_validation_gradients(val_dataloader, max_val_samples)

    def get_validation_gradients(self, dataloader_or_dataset, max_val_samples=None):
        """Compute gradients on validation dataset for influence computation
        
        Args:
            dataloader_or_dataset: Validation dataloader or dataset
            max_val_samples: Maximum number of validation samples to use. If None, uses all samples.
        """
        
        # Save current training state
        was_training = self.training
        self.eval()
        self.zero_grad()
        
        # Handle both dataset and dataloader inputs
        val_dataloader = dataloader_or_dataset
        
        total_samples = 0
        samples_processed = 0
        
        print(f"Computing validation gradients over {len(val_dataloader)} batches...")
        if max_val_samples is not None:
            print(f"Limiting to maximum {max_val_samples} validation samples")
        print(f"NOTE: Validation dataset indices will NOT interfere with training influence scores")
        
        # Collect all validation data first for single backward pass
        all_batches = []
        
        for batch_idx, val_batch in enumerate(val_dataloader):
            try:
                # IMPORTANT: Remove dataset_index from validation batch to prevent contamination
                if self.cache_val_batch is not None:
                    val_batch_clean = self.cache_val_batch
                    print("Using cached val batch --------------------------------")
                else:
                    val_batch_clean = {k: v for k, v in val_batch.items() if k != 'dataset_index'}
                    self.cache_val_batch = val_batch_clean
                # Move tensors to device
                val_batch_clean = {
                    k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in val_batch_clean.items()
                }
                
                # Get batch size for this batch
                batch_size = (
                    val_batch_clean["sample_id"].max(dim=1).values.sum().item() 
                    if "sample_id" in val_batch_clean and val_batch_clean["sample_id"].dim() > 1
                    else val_batch_clean["target"].shape[0]
                )
                
                # Check if we've reached the limit
                if max_val_samples is not None and samples_processed + batch_size > max_val_samples:
                    # Only take what we need from this batch
                    remaining_samples = max_val_samples - samples_processed
                    if remaining_samples <= 0:
                        break
                    
                    # Truncate the batch to only take remaining_samples
                    for key in val_batch_clean:
                        if torch.is_tensor(val_batch_clean[key]) and val_batch_clean[key].dim() > 0:
                            val_batch_clean[key] = val_batch_clean[key][:remaining_samples]
                    
                    batch_size = remaining_samples
                
                # Store the batch for later processing
                all_batches.append(val_batch_clean)
                
                total_samples += batch_size
                samples_processed += batch_size
                
                # Clear intermediate variables to free memory
                del val_batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(val_dataloader)} validation batches")
                
                # Check if we've reached the limit
                if max_val_samples is not None and samples_processed >= max_val_samples:
                    print(f"Reached maximum validation samples limit: {max_val_samples}")
                    break
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        if total_samples == 0:
            print("Warning: No validation samples processed")
            return {}
        
        print(f"Collected {total_samples} validation samples, computing gradients...")
        
        # Now compute the total loss and do a single backward pass
        try:
            # Compute total loss across all collected data with gradients enabled
            total_loss = 0.0
            num_batches = len(all_batches)
            
            for i, val_batch_clean in enumerate(all_batches):
                # Forward pass with gradient computation enabled
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
                    
                    # Add to total loss
                    total_loss += batch_loss
            
            # Average the loss
            total_loss = total_loss / num_batches
            
            # Single backward pass
            total_loss.backward()
            
            # Extract gradients
            val_gradients = {}
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    val_gradients[name] = param.grad.clone().detach()
            
            # Clear gradients and restore training state
            self.zero_grad()
            if was_training:
                self.train()
            
            print(f"Computed validation gradients for {len(val_gradients)} parameters over {total_samples} samples")
            
            return val_gradients
            
        except Exception as e:
            print(f"Error in single backward pass: {e}")
            print("Falling back to original method...")
            
            # Fallback: Clear gradients and restore training state
            self.zero_grad()
            if was_training:
                self.train()
            
            # Return empty dict to indicate failure
            return {}

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