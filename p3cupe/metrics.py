
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.ph66_mapper import phoneme_mapped_index, phoneme_groups_index



class MetricsCalculator5(nn.Module):
    '''
    Metrics calculator. PER is implemented.
    '''

    def __init__(self, blank_class, total_classes, total_entries, ph_seq_max, max_spec_len, blank_group=None, total_groups=1, mode='greedy', device='cpu', use_pred_lengths=True, light_metrics=True):
        super().__init__()

        # `light_metrics` skips the four sequence metrics that are Python loops over the
        # whole val set AND are not diagnostic for boundary work:
        #   transition_sharpness, alignment_decisiveness, duration_consistency, ordering
        # Sharpness and decisiveness in particular cannot distinguish the good case from
        # the bad one here - the partition CE is itself a confidence-maximising objective,
        # so both rise whether the model is sharpening the right thing or the wrong thing.
        # PER, GER, coverage and all the blank/CE stats are kept.
        self.light_metrics = bool(light_metrics)
        
        assert(blank_class < total_classes) # expect blank class to be the last class, to be included in the total_classes

        self.blank_class_id = blank_class
        self.blank_group_id = blank_group
        self.total_classes = int(total_classes)
        self.total_groups = int(total_groups)
        assert (self.total_groups < self.total_classes)
        self.device = device
        self.mode = mode
        
        # Add simple decoder alongside existing ones
        if mode == 'greedy':
            self.decoder = GreedyDecoder()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['greedy', 'beam', 'simple']") # removed beam and simple. Find in older revisions if needed.
            
        self.use_pred_lengths = use_pred_lengths # Use a known prediction lengths for decoding, True by default. prediction length is sent by the master module calculated based on the frames per module.
        
        self.phoneme_labels = phoneme_mapped_index
        if(len(self.phoneme_labels) != total_classes):
            print( "Make sure hyperparameter `phoneme_classes` matches the number of phonemes in the phoneme_mapped_index (? 66 maybe - could change)")
            self.phoneme_labels = [f"{i:02d}" for i in range(total_classes)]
        self.group_labels = phoneme_groups_index
        if (self.total_groups > 1):
            assert(len(self.group_labels) == total_groups), "Make sure hyperparameter `phoneme_groups` matches the number of phoneme groups in the phoneme_groups_index (? 19 maybe - could change)"

        self.total_entries = int(total_entries)
        self.ph_seq_max = int(ph_seq_max) # +10 for boundaries, has to be same in WordPhonesAudio
        self.max_spec_len = int(max_spec_len)
        self.preds_mem = None
        self.pred_grps_mem = None
        self.preds_lengths_mem = None
        self.targets_mem = None
        self.target_grps_mem = None
        self.target_lengths_mem = None

        self.reset()

    def reset(self):
        if (self.total_entries > 1e6): #2391337
            typex = torch.float16
        else: typex = torch.float32
        self.preds_mem = torch.zeros(self.total_entries, self.max_spec_len, self.total_classes, device=self.device, dtype=typex, requires_grad=False)
        self.preds_lengths_mem = torch.zeros(self.total_entries, device=self.device, dtype=torch.long, requires_grad=False)
        self.targets_mem = torch.zeros(self.total_entries, self.ph_seq_max, device=self.device, dtype=torch.long, requires_grad=False)
        self.target_lengths_mem = torch.zeros(self.total_entries, device=self.device, dtype=torch.long, requires_grad=False)

        if (self.total_groups > 1):
            self.pred_grps_mem = torch.zeros(self.total_entries, self.max_spec_len, self.total_groups, device=self.device, dtype=typex, requires_grad=False)
            self.target_grps_mem = torch.zeros(self.total_entries, self.ph_seq_max, device=self.device, dtype=torch.long, requires_grad=False)
        
        self.bnd_mem = torch.zeros(self.total_entries, self.ph_seq_max+10, device=self.device, dtype=torch.long, requires_grad=False)
        # Predicted onsets from the batched Viterbi forced alignment (same units and
        # padding convention as bnd_mem). Filled by add_batch when the trainer runs
        # the aligner; left at zero otherwise, in which case compute_boundary_metrics
        # falls back to the old argmax-derived boundaries.
        self.pred_bnd_mem = torch.zeros(self.total_entries, self.ph_seq_max+10, device=self.device, dtype=torch.long, requires_grad=False)
        self.has_pred_bnd = False
        self.idx = 0

    def clear_mem(self):
        del self.preds_mem
        del self.pred_grps_mem
        del self.preds_lengths_mem
        del self.targets_mem
        del self.target_grps_mem
        del self.target_lengths_mem
        del self.bnd_mem
        del self.pred_bnd_mem
        self.idx = 0

    def add_batch(self, preds, targets, target_lengths, grp_preds=None, grp_targets=None, pred_lengths=None, target_boundaries=None, pred_boundaries=None):
        batch_size = preds.shape[0]
        if self.idx + batch_size > self.total_entries:
            raise ValueError(f"Buffer overflow: Cannot add batch of size {batch_size}")
        


        # Ensure everything is on the correct device
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        target_lengths = target_lengths.to(self.device)

        '''
        if grp_preds is not None:
            grp_preds = grp_preds.to(self.device)
        if grp_targets is not None:
            grp_targets = grp_targets.to(self.device)

        if pred_lengths is not None:
            pred_lengths = pred_lengths.to(self.device)
        
        if target_boundaries is not None:
            target_boundaries = target_boundaries.to(self.device)
        #self._validate_log_probs(preds)
        '''

        # Validate shapes
        assert preds.shape[-1] == self.total_classes, "Invalid number of classes in predictions"
        assert targets.shape[-1] <= self.ph_seq_max, "Target sequence too long"
        
        for i in range(batch_size):
            #print(self.preds_mem[self.idx].shape, preds[i].shape)
            
            self.preds_mem[self.idx] = preds[i]
            self.targets_mem[self.idx] = targets[i]
            self.target_lengths_mem[self.idx] = target_lengths[i]

            
            if (self.total_groups > 1):
                self.pred_grps_mem[self.idx] = grp_preds[i]
                self.target_grps_mem[self.idx] = grp_targets[i]

            if (self.use_pred_lengths):
                self.preds_lengths_mem[self.idx] = pred_lengths[i]

            if target_boundaries is not None:
                self.bnd_mem[self.idx] = target_boundaries[i]

            if pred_boundaries is not None:
                self.pred_bnd_mem[self.idx] = pred_boundaries[i]
                self.has_pred_bnd = True

            self.idx += 1

    def _validate_log_probs(self, log_probs):
        """Validate that inputs are log probabilities with some tolerance"""
        with torch.no_grad():
            # Convert to probabilities
            probs = torch.exp(log_probs)
            # Sum along class dimension
            probs_sum = probs.sum(dim=-1)
            # Check if sums are close to 1 with a reasonable tolerance
            is_valid = torch.allclose(probs_sum, torch.ones_like(probs_sum), rtol=1e-4, atol=1e-4)
            return is_valid

    
    
    @staticmethod
    def _match_boundaries_monotonic(pred, true, match_tol):
        """
        Monotonic one-to-one matching between two ascending boundary lists.

        Mirrors the offline scorer (`modules/aligner/alignment_tests.py`): match
        first, threshold afterwards. A greedy two-pointer walk keeps the matching
        order-preserving and one-to-one, so a single predicted boundary can no
        longer be credited against several true boundaries.

        Args:
            pred: ascending list of predicted boundary frames
            true: ascending list of true boundary frames
            match_tol: maximum distance (frames) at which a pair may be matched at all

        Returns:
            List of absolute deviations (frames) for the matched pairs.
        """
        deviations = []
        i = j = 0
        while i < len(pred) and j < len(true):
            diff = pred[i] - true[j]
            if abs(diff) <= match_tol:
                deviations.append(abs(diff))
                i += 1
                j += 1
            elif diff < 0:
                i += 1   # predicted boundary with no true counterpart (insertion)
            else:
                j += 1   # true boundary the model missed
        return deviations

    def compute_boundary_metrics(self, tolerances=(1, 2, 3), match_tol=10):
        """
        Calculate metrics for phoneme boundary detection performance.

        Predicted boundaries come from the **batched Viterbi forced alignment**
        (`pred_bnd_mem`, filled by the trainer) - the onset frame of each aligned
        target phoneme. This is the same quantity the offline eval scores, so these
        numbers can be read as a proxy for it. Before 2026-07-29 the predicted
        boundaries were derived by argmax over frames at a fixed 3-frame tolerance,
        which measured something different from the offline eval and therefore could
        not confirm that a boundary-sharpening change had worked (§5 of the plan).
        The argmax path is kept only as a fallback for when the trainer did not run
        the aligner.

        Matching is done once at `match_tol`, then thresholded at each tolerance -
        the same order of operations as the offline scorer.

        Args:
            tolerances: frame tolerances to report F1 at. At ~16ms per frame these
                are roughly 16/32/48ms.
            match_tol: frames within which a pair may be matched at all. Deviations
                over matched pairs are reported as `boundary_abs_err`, the in-training
                analogue of the offline `abs_err_mean`.

        Returns:
            dict with `boundary_abs_err` (mean frames over matched pairs),
            `boundary_f1_t{tol}` for each tolerance, `boundary_recall` /
            `boundary_precision` at the middle tolerance, and
            `boundary_duration_deviation`.
        """
        with torch.no_grad():
            use_viterbi = bool(self.has_pred_bnd)

            total_true = 0
            total_pred = 0
            all_deviations = []
            duration_deviations = []

            desc = "boundary_metrics" + ("" if use_viterbi else " (argmax fallback)")
            for i in tqdm(range(0, self.idx), desc=desc, leave=False):
                seq_len = self.preds_lengths_mem[i] if self.use_pred_lengths else self.preds_mem[i].shape[0]
                target_boundaries = self.bnd_mem[i]

                # Skip empty samples
                if seq_len == 0 or target_boundaries[0] == 0:
                    continue

                valid_targets = [b.item() for b in target_boundaries if b > 0]

                if use_viterbi:
                    # Onsets of the forced-aligned phonemes, already ascending.
                    pred_boundaries = [b.item() for b in self.pred_bnd_mem[i] if b > 0]
                else:
                    # Fallback: frames where the argmax phoneme changes.
                    pred_phonemes = torch.argmax(self.preds_mem[i][:seq_len], dim=1)
                    pred_boundaries = []
                    prev_phoneme = None
                    for t, phoneme in enumerate(pred_phonemes):
                        phoneme_idx = phoneme.item()
                        if phoneme_idx != prev_phoneme and phoneme_idx != self.blank_class_id and t > 0:
                            pred_boundaries.append(t)
                        prev_phoneme = phoneme_idx

                total_true += len(valid_targets)
                total_pred += len(pred_boundaries)

                all_deviations.extend(
                    self._match_boundaries_monotonic(pred_boundaries, valid_targets, match_tol)
                )

                # Duration deviation: how well the gaps between consecutive
                # boundaries match, as a fraction of the true duration.
                if len(valid_targets) > 1 and len(pred_boundaries) > 1:
                    target_durations = [valid_targets[k+1] - valid_targets[k] for k in range(len(valid_targets)-1)]
                    pred_durations = [pred_boundaries[k+1] - pred_boundaries[k] for k in range(len(pred_boundaries)-1)]
                    for k in range(min(len(target_durations), len(pred_durations))):
                        duration_deviations.append(
                            abs(target_durations[k] - pred_durations[k]) / max(target_durations[k], 1)
                        )

            results = {}

            if all_deviations:
                dev = torch.tensor(all_deviations, dtype=torch.float32)
                results['boundary_abs_err'] = dev.mean().item()
                results['boundary_abs_err_median'] = dev.median().item()
            else:
                dev = None
                results['boundary_abs_err'] = float('inf')
                results['boundary_abs_err_median'] = float('inf')

            mid_tol = tolerances[len(tolerances) // 2]
            for tol in tolerances:
                correct = int((dev <= tol).sum().item()) if dev is not None else 0
                precision = correct / max(total_pred, 1)
                recall = correct / max(total_true, 1)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                results[f'boundary_f1_t{tol}'] = f1
                if tol == mid_tol:
                    results['boundary_precision'] = precision
                    results['boundary_recall'] = recall

            results['boundary_duration_deviation'] = (
                sum(duration_deviations) / len(duration_deviations) if duration_deviations else float('inf')
            )
            results['boundary_detection_ratio'] = total_pred / max(total_true, 1)

            return results


    def compute_sequential_metrics(self, do_groups=True):
        """
        Compute comprehensive metrics for evaluating phoneme recognition quality.
        
        This function calculates multiple metrics that evaluate different aspects of
        phoneme recognition performance beyond simple accuracy, including confidence,
        coverage, consistency, and ordering.
        
        Returns:
            dict: Dictionary containing the following metrics:
                - decisiveness (float): Average maximum probability across all frames,
                ranging from 0 to 1. Higher values indicate more confident predictions.

                - phoneme_coverage (float): Percentage of target phonemes that appear in
                the predictions (regardless of position), ranging from 0 to 1.

                - duration_consistency (float): Consistency of phoneme durations across
                different instances of the same phoneme, ranging from 0 to 1. Higher
                values indicate more consistent duration modeling.

                - transition_sharpness (float): Measure of how decisively the model
                transitions between phonemes, based on probability distribution entropy
                at transition points. Higher values (0-1) indicate sharper transitions.

                - phoneme_ordering (float): How well the model preserves the correct
                sequence of phonemes, calculated using longest common subsequence.
                Ranges from 0 to 1, with higher values indicating better ordering.

                - PER (float): Phoneme Error Rate, calculated as 
                (substitutions + deletions + insertions) / total reference phonemes.
                Lower values indicate better overall performance.
        """
    
        # Initialize metrics dictionary
        metrics = {}

        with torch.no_grad():

            # Calculate phoneme coverage
            coverage_score = self.calculate_phoneme_coverage()
            metrics['seq_coverage_p'] = coverage_score

            # The four below are Python loops over the whole val set and are NOT diagnostic
            # for boundary work (see `light_metrics` in __init__). Skipped by default.
            if not self.light_metrics:
                # doesn't check if the decision is right or wrong, just how confident the model is
                metrics['seq_decisiveness_p'] = self.calculate_alignment_decisiveness(groups=False)
                metrics['seq_duration_p'] = self.calculate_duration_consistency(groups=False)
                metrics['seq_sharpness_p'] = self.calculate_transition_sharpness(groups=False)
                metrics['seq_ordering_p'] = self.calculate_phoneme_ordering_accuracy()

                if (do_groups and self.total_groups > 1):
                    metrics['seq_decisiveness_g'] = self.calculate_alignment_decisiveness(groups=True)
                    metrics['seq_duration_g'] = self.calculate_duration_consistency(groups=True)
                    metrics['seq_sharpness_g'] = self.calculate_transition_sharpness(groups=True)


            per = self.calculate_per(groups=False)
            metrics['PER'] = per

            if (do_groups and self.total_groups > 1):
                per_grp = self.calculate_per(groups=True)
                metrics['GER'] = per_grp

        return metrics



        

    
    def calculate_per(self, groups=False):
        '''
        Calculate Phoneme Error Rate (PER) for the stored predictions and targets. 
       
        
        Returns: 'per': float, the overall PER
        '''
        

        with torch.no_grad():
            #print("\ncalculate_per, Total_classes:", self.total_classes-1, " Total sequences:",self.idx)
            # Decode predictions using selected decoder

            ''' INFO
            # Metrics are calculated on the memory for predictions and targets.
            # self.preds_mem = torch.zeros(self.total_entries, self.max_spec_len, self.total_classes, device=self.device)
            # self.preds_lengths_mem = torch.zeros(self.total_entries, device=self.device, dtype=torch.long)
            '''

            if self.use_pred_lengths:
                if groups:
                    decoded_preds = self.decoder.decode(self.pred_grps_mem, self.blank_group_id, self.preds_lengths_mem)
                else:   
                    decoded_preds = self.decoder.decode(self.preds_mem, self.blank_class_id, self.preds_lengths_mem)
            else:
                if groups:
                    decoded_preds = self.decoder.decode(self.pred_grps_mem, self.blank_group_id)
                else:
                    decoded_preds = self.decoder.decode(self.preds_mem, self.blank_class_id)
            


            # Calculate PER for each utterance
            batch_errors = []
            
            batch_lengths = []
            
            for i in tqdm(range(0, self.idx), desc="calculate_per", leave=False):
                pred_seq = decoded_preds[i]
                if groups:
                    true_seq = self.target_grps_mem[i][:self.target_lengths_mem[i]].tolist()
                else:
                    true_seq = self.targets_mem[i][:self.target_lengths_mem[i]].tolist()
                
                

                if self.mode == 'simple':
                    # For simple mode, use normalized Levenshtein distance
                    error = self.decoder.calculate_distance(pred_seq, true_seq)
                    ref_length = 1.0  # Since the error is already normalized
                else:
                    # For other modes, use detailed edit operations
                    substitutions, deletions, insertions = self._get_edit_stats(pred_seq, true_seq)
                    error = substitutions + deletions + insertions

                    ref_length = len(true_seq)

                
                batch_errors.append(error)
                batch_lengths.append(ref_length)
                
            # Calculate overall PER
            if self.mode == 'simple':
                per = sum(batch_errors) / len(batch_errors) if batch_errors else 1.0
            else:
                total_errors = sum(batch_errors)
                total_phones = sum(batch_lengths)
                per = total_errors / total_phones if total_phones > 0 else 1.0
            # Calculate individual PERs
           # pers = batch_errors if self.mode == 'simple' else [e/l if l > 0 else 1.0 for e, l in zip(batch_errors, batch_lengths)]
            
            
            return per


    def _get_edit_stats(self, pred_seq, true_seq):
        """Calculate detailed statistics about the edits needed."""
        m, n = len(pred_seq), len(true_seq)
        dp = torch.zeros((m + 1, n + 1), device=self.device)
        
        # Initialize
        dp[0] = torch.arange(n + 1, device=self.device)  # Pure insertions
        dp[:, 0] = torch.arange(m + 1, device=self.device)  # Pure deletions
        
        # Operation traceback storage
        ops = torch.zeros((m + 1, n + 1), device=self.device, dtype=torch.long)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_seq[i-1] == true_seq[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                    ops[i, j] = 0  # Correct
                else:
                    # Minimum of: deletion, insertion, substitution
                    candidates = torch.tensor([dp[i-1, j] + 1,    # deletion
                                            dp[i, j-1] + 1,    # insertion
                                            dp[i-1, j-1] + 1], # substitution
                                           device=self.device)
                    dp[i, j] = candidates.min()
                    ops[i, j] = candidates.argmin() + 1
        
        # Count operations
        i, j = m, n
        substitutions = deletions = insertions = 0
        
        while i > 0 and j > 0:
            if ops[i, j] == 0:  # Correct
                i -= 1
                j -= 1
            elif ops[i, j] == 1:  # Deletion
                deletions += 1
                i -= 1
            elif ops[i, j] == 2:  # Insertion
                insertions += 1
                j -= 1
            else:  # Substitution
                substitutions += 1
                i -= 1
                j -= 1
                
        # Handle remaining ops
        deletions += i
        insertions += j
        
        return substitutions, deletions, insertions
    


        
        
    def calculate_alignment_decisiveness(self, groups=False):
        """
        Calculate the average decisiveness of the model in its phoneme predictions.
        Higher values indicate more confident alignments.
        
        Returns: float (0-1), where higher is better
        """
        with torch.no_grad():
            total_confidence = 0.0
            total_frames = 0
            
            for i in tqdm(range(0, self.idx), desc="alignment_decisiveness", leave=False):
                # Get predictions
                preds = self.pred_grps_mem[i] if (groups) else self.preds_mem[i]
                seq_len = self.preds_lengths_mem[i] if self.use_pred_lengths else preds.shape[0]
                
                # Skip empty samples
                if seq_len == 0:
                    continue
                
                # Convert log probs to probabilities
                probs = torch.exp(preds[:seq_len])
                
                # Get maximum probability at each frame as a decisiveness score
                frame_confidence = torch.max(probs, dim=1)[0]
                
                # Calculate average decisiveness for this sequence
                total_confidence += frame_confidence.sum().item()
                total_frames += seq_len
            
            # Return average decisiveness across all frames
            avg_ = (total_confidence / total_frames).cpu().detach().item()
            return avg_ if total_frames > 0 else 0.0

    def calculate_phoneme_coverage(self):
        """
        Calculate what percentage of target phonemes are detected in the predictions.
        This measures if the model is capturing all expected phonemes.
        
        Returns: float (0-1), where higher is better
        """
        with torch.no_grad():
            total_coverage = 0.0
            total_sequences = 0
            
            for i in tqdm(range(0, self.idx), desc="phoneme_coverage", leave=False):
                # Get predictions and targets
                preds = self.preds_mem[i]
                target_seq = self.targets_mem[i][:self.target_lengths_mem[i]]
                seq_len = self.preds_lengths_mem[i] if self.use_pred_lengths else preds.shape[0]
                
                # Skip empty samples
                if seq_len == 0 or len(target_seq) == 0:
                    continue
                
                # Get most likely phoneme for each frame
                pred_phonemes = torch.argmax(preds[:seq_len], dim=1)
                
                # Count unique phonemes in prediction and target
                unique_targets = set(target_seq.cpu().tolist())
                unique_preds = set(pred_phonemes.cpu().tolist())
                
                # Remove blank token from predictions
                if self.blank_class_id in unique_preds:
                    unique_preds.remove(self.blank_class_id)
                
                # Calculate coverage (percentage of target phonemes found in predictions)
                if len(unique_targets) > 0:
                    coverage = len(unique_targets.intersection(unique_preds)) / len(unique_targets)
                    total_coverage += coverage
                    total_sequences += 1
            
            # Return average coverage across all sequences
            return total_coverage / total_sequences if total_sequences > 0 else 0.0

    def calculate_duration_consistency(self, groups=False):
        """
        Measure how consistent the model is with phoneme durations.
        Looks at variance in duration across instances of the same phoneme.
        
        Returns: float (0-1), where higher is better (more consistent)
        """
        with torch.no_grad():
            # Track durations for each phoneme type
            phoneme_durations = {}
            blank_id =self.blank_group_id if groups else  self.blank_class_id 
            for i in tqdm(range(0, self.idx), desc="duration_consistency", leave=False):
                # Get predictions
                preds = self.pred_grps_mem[i] if (groups) else self.preds_mem[i]
                seq_len = self.preds_lengths_mem[i] if self.use_pred_lengths else preds.shape[0]
                
                # Skip empty samples
                if seq_len == 0:
                    continue
                
                # Get most likely phoneme for each frame
                pred_phonemes = torch.argmax(preds[:seq_len], dim=1).cpu().tolist() if groups else torch.argmax(preds[:seq_len], dim=1).cpu().tolist()
                
                # Count consecutive frames for each phoneme
                current_phoneme = None
                current_duration = 0
                
                for phoneme in pred_phonemes:
                    if phoneme == current_phoneme:
                        current_duration += 1
                    else:
                        # Store duration of previous phoneme
                        if current_phoneme is not None and current_phoneme != blank_id:
                            if current_phoneme not in phoneme_durations:
                                phoneme_durations[current_phoneme] = []
                            phoneme_durations[current_phoneme].append(current_duration)
                        
                        # Reset for new phoneme
                        current_phoneme = phoneme
                        current_duration = 1
                
                # Add final phoneme
                if current_phoneme is not None and current_phoneme != blank_id:
                    if current_phoneme not in phoneme_durations:
                        phoneme_durations[current_phoneme] = []
                    phoneme_durations[current_phoneme].append(current_duration)
            
            # Calculate coefficient of variation (CV) for each phoneme's durations
            # CV = standard deviation / mean (normalized variance)
            total_consistency = 0.0
            phoneme_count = 0
            
            for phoneme, durations in phoneme_durations.items():
                if len(durations) > 1:  # Need at least 2 samples to calculate variance
                    mean_duration = sum(durations) / len(durations)
                    variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
                    
                    # Calculate coefficient of variation
                    cv = (variance ** 0.5) / mean_duration if mean_duration > 0 else float('inf')
                    
                    # Convert to consistency score (1 = perfect consistency, 0 = high variability)
                    # Using 1/(1+cv) to map [0, inf) to (0, 1]
                    consistency = 1.0 / (1.0 + cv)
                    
                    total_consistency += consistency
                    phoneme_count += 1
            
            # Return average consistency across all phonemes
            return total_consistency / phoneme_count if phoneme_count > 0 else 0.0

    def calculate_transition_sharpness(self, groups=False):
        """
        Measure how decisive the transitions are between phonemes.
        Sharp transitions suggest more precise boundary detection.
        
        Returns: float (0-1), where higher is better
        """
        with torch.no_grad():
            total_sharpness = 0.0
            total_transitions = 0
            total_categories = self.total_classes if not groups else self.total_groups
            max_entropy = torch.log(torch.tensor(float(total_categories), device=self.device))

            for i in tqdm(range(0, self.idx), desc="transition_sharpness", leave=False):
                # Get predictions
                preds = self.pred_grps_mem[i] if (groups) else self.preds_mem[i]
                seq_len = self.preds_lengths_mem[i] if self.use_pred_lengths else preds.shape[0]
                
                # Skip empty or very short samples
                if seq_len <= 2:
                    continue
                
                # Convert to probabilities
                probs = torch.exp(preds[:seq_len])
                
                # Get most likely phoneme for each frame
                pred_phonemes = torch.argmax(probs, dim=1)
                
                # Find transitions (where phoneme changes)
                transitions = []
                for t in range(1, seq_len):
                    if pred_phonemes[t] != pred_phonemes[t-1]:
                        transitions.append(t)
                
                # Calculate sharpness around each transition
                for t in transitions:
                    if 0 < t < seq_len - 1:  # Ensure we can look at before/after frames
                        # Get probabilities before, at, and after transition
                        before_probs = probs[t-1]
                        at_probs = probs[t]
                        after_probs = probs[t+1]
                        
                        # Calculate entropy (lower entropy = sharper distribution)
                        before_entropy = -(before_probs * torch.log(before_probs + 1e-10)).sum()
                        at_entropy = -(at_probs * torch.log(at_probs + 1e-10)).sum()
                        after_entropy = -(after_probs * torch.log(after_probs + 1e-10)).sum()
                        
                        # Normalize by maximum possible entropy
                        #max_entropy = torch.log(torch.tensor(float(total_categories), device=before_probs.device))
                        avg_entropy = (before_entropy + at_entropy + after_entropy) / (3 * max_entropy)
                        
                        # Convert to sharpness score (1 - normalized entropy)
                        # 1 = perfectly sharp, 0 = completely uniform
                        sharpness = 1.0 - avg_entropy.item()
                        
                        total_sharpness += sharpness
                        total_transitions += 1
            
            # Return average sharpness across all transitions
            return total_sharpness / total_transitions if total_transitions > 0 else 0.0


    def calculate_phoneme_ordering_accuracy(self):
        """
        Measure how well the model preserves the expected ordering of phonemes.
        
        Returns: float (0-1), where higher is better
        """
        with torch.no_grad():
            total_accuracy = 0.0
            total_sequences = 0
            
            for i in tqdm(range(0, self.idx), desc="ordering_accuracy", leave=False):
                # Get predictions and targets
                preds = self.preds_mem[i]
                target_seq = self.targets_mem[i][:self.target_lengths_mem[i]]
                seq_len = self.preds_lengths_mem[i] if self.use_pred_lengths else preds.shape[0]
                
                # Skip empty samples
                if seq_len == 0 or len(target_seq) == 0:
                    continue
                
                # Get most likely phoneme for each frame
                pred_phonemes = torch.argmax(preds[:seq_len], dim=1)
                
                # Extract predicted phoneme sequence (removing duplicates and blanks)
                pred_sequence = []
                prev_phoneme = None
                for ph in pred_phonemes:
                    ph_idx = ph.item()
                    if ph_idx != prev_phoneme and ph_idx != self.blank_class_id:
                        pred_sequence.append(ph_idx)
                        prev_phoneme = ph_idx
                
                # Convert target sequence to list
                target_list = target_seq.cpu().tolist()
                
                # Calculate longest common subsequence (LCS)
                lcs_length = self._longest_common_subsequence(pred_sequence, target_list)
                
                # Calculate accuracy as ratio of LCS to target length
                accuracy = lcs_length / len(target_list) if target_list else 0.0
                
                total_accuracy += accuracy
                total_sequences += 1
            
            # Return average accuracy across all sequences
            return total_accuracy / total_sequences if total_sequences > 0 else 0.0

    def _longest_common_subsequence(self, seq1, seq2):
        """
        Calculate the length of the longest common subsequence between two sequences.
        This measures how well the order of elements is preserved.
        
        Args:
            seq1: First sequence (list of phoneme indices)
            seq2: Second sequence (list of phoneme indices)
            
        Returns:
            Length of the longest common subsequence
        """
        # Create a table to store LCS lengths
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the table using dynamic programming
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Return the length of the LCS
        return dp[m][n]


    
    def _get_alignment_v1(self, pred_seq, true_seq):
        """
        Calculate optimal alignment between predicted and true sequences.
        Returns list of (pred_idx, true_idx) pairs.
        """
        m, n = len(pred_seq), len(true_seq)
        device = pred_seq.device
        
        # Initialize dp matrix with infinity and backtrack matrix
        dp = torch.full((m + 1, n + 1), float('inf'), device=device)
        backtrack = torch.zeros((m + 1, n + 1), device=device, dtype=torch.long)
        
        # Initialize first row and column more efficiently
        dp[0] = torch.arange(n + 1, device=device)
        dp[:, 0] = torch.arange(m + 1, device=device)
        backtrack[0, 1:] = 1
        backtrack[1:, 0] = 2
        
        # Pre-compute equality matrix
        pred_expanded = pred_seq.unsqueeze(1).expand(m, n)
        true_expanded = true_seq.unsqueeze(0).expand(m, n)
        sub_costs = (pred_expanded != true_expanded).float()
        
        # Fill dp matrix more efficiently
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate all costs at once
                costs = torch.tensor([
                    dp[i-1, j-1] + sub_costs[i-1, j-1],  # substitution/match
                    dp[i, j-1] + 1,                      # insertion
                    dp[i-1, j] + 1                       # deletion
                ], device=device)
                
                dp[i, j], backtrack[i, j] = costs.min(0)
        
        # Backtrack (keep this part the same for exact output matching)
        alignment = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and backtrack[i, j] == 0:
                alignment.append((pred_seq[i-1], true_seq[j-1]))
                i -= 1
                j -= 1
            elif j > 0 and backtrack[i, j] == 1:
                alignment.append((None, true_seq[j-1]))
                j -= 1
            else:
                alignment.append((pred_seq[i-1], None))
                i -= 1
        
        return list(reversed(alignment))
    
    def calculate_confusion_matrix(self, save_path=None, n_batches=0, groups=False):
        """
        Memory-optimized version of confusion matrix calculation
        """
        total_categories = self.total_classes if (not groups) else self.total_groups
        unmatched_preds =  torch.zeros((total_categories,), device=self.device, dtype=torch.long)
        unmatched_trues = torch.zeros((total_categories,), device=self.device, dtype=torch.long)
        with torch.no_grad():
            assert(self.total_classes-1 == len(phoneme_mapped_index) - 1)
            
            total_rows = total_categories-1
            matrix = torch.zeros((total_rows, total_rows), device=self.device)
            matrix_probs = torch.zeros((total_rows, total_rows), device=self.device)
            
            # Decode predictions in smaller chunks to reduce memory usage
            chunk_size = 100  # Reduced from 1000 to prevent memory spikes
            total_chunks = (self.idx + chunk_size - 1) // chunk_size
            if (n_batches > 0) and (total_chunks>n_batches): total_chunks = n_batches
            
            for chunk_idx in tqdm(range(total_chunks), desc="calculate_confusion", leave=False):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, self.idx)
                
                # Process predictions in chunks
                if (groups):
                    chunk_preds = self.decoder.decode(
                        self.pred_grps_mem[start_idx:end_idx],
                        self.blank_group_id,
                        self.preds_lengths_mem[start_idx:end_idx] if self.use_pred_lengths else None
                    )
                else:
                    chunk_preds = self.decoder.decode(
                        self.preds_mem[start_idx:end_idx],
                        self.blank_class_id,
                        self.preds_lengths_mem[start_idx:end_idx] if self.use_pred_lengths else None
                    )
                
                # Process alignments immediately for this chunk
                for i, pred_seq in enumerate(chunk_preds):
                    abs_idx = start_idx + i
                    pred_seq = torch.tensor(pred_seq, device=self.device, dtype=torch.long)
                    true_seq = self.targets_mem[abs_idx][:self.target_lengths_mem[abs_idx]] if (not groups) else self.target_grps_mem[abs_idx][:self.target_lengths_mem[abs_idx]]
                    
                    # Process alignment one sequence at a time
                    alignment = self._get_alignment_v1(pred_seq, true_seq)
                    
                    for pred_idx, true_idx in alignment:
                        if pred_idx is not None and true_idx is not None:
                            if (0 <= true_idx < total_rows and 0 <= pred_idx < total_rows):
                                matrix[true_idx, pred_idx] += 1
                                
                                # Calculate probabilities one at a time to save memory
                                if self.use_pred_lengths:
                                    valid_length = self.preds_lengths_mem[abs_idx]
                                    log_prob = self.preds_mem[abs_idx, :valid_length, true_idx] if (not groups) else (self.pred_grps_mem[abs_idx, :valid_length, true_idx])
                                else:
                                    log_prob = self.preds_mem[abs_idx, :, true_idx] if (not groups) else (self.pred_grps_mem[abs_idx, :, true_idx])
                                
                                prob = torch.exp(log_prob.mean())
                                matrix_probs[true_idx, pred_idx] += prob.item()
                        elif (pred_idx is not None):
                            unmatched_preds[pred_idx] += 1
                        elif (true_idx is not None):
                            unmatched_trues[true_idx] += 1
                # Clear CUDA cache after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            
            # Normalize probabilities
            row_sums = matrix_probs.sum(dim=1, keepdim=True)
            matrix_probs = matrix_probs / torch.maximum(row_sums, torch.tensor(1e-8, device=self.device))
            
            # Save results if path provided
            if save_path and (not groups):
                self._save_matrices(save_path, matrix, matrix_probs, total_rows)
            
            return matrix, matrix_probs, unmatched_preds, unmatched_trues

    def _save_matrices(self, save_path, matrix, matrix_probs, total_rows):
        """
        Separate method for saving matrices to reduce memory usage in main function
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Move to CPU and save one at a time
        matrix_cpu = matrix.cpu()
        torch.save(matrix_probs.cpu(), f"{save_path}.pt")
        
        try:
            # Prepare labels
            if hasattr(self, 'phoneme_labels') and self.phoneme_labels is not None:
                index_to_label = {v: k for k, v in self.phoneme_labels.items()}
                labels = [index_to_label.get(i, str(i)) for i in range(total_rows)]
            else:
                labels = [str(i) for i in range(total_rows)]
            
            # Save CSV
            matrix_np = matrix_cpu.numpy()
            with open(f"{save_path}.csv", 'w') as f:
                f.write('true\\pred,' + ','.join(labels) + '\n')
                for i in range(total_rows):
                    row_values = ','.join(map(str, matrix_np[i].astype(int)))
                    f.write(f"{labels[i]},{row_values}\n")
                    
        except Exception as e:
            print(f"Error saving confusion matrix to CSV: {str(e)}")
            raise




    
    
class GreedyDecoder:
    def __init__(self):
        pass

    def decode(self, preds, blank_id, pred_lengths=None):
        # Take argmax over class dimension (dim=2) to get the most probable token at each time step
        argmax_preds = torch.argmax(preds, dim=2)
        
        # Collapsing repeated tokens and removing blanks
        decoded_preds = []
        for n in range(argmax_preds.shape[0]):
            collapsed_seq = []
            prev_token = blank_id
            seq = argmax_preds[n]
            seq_len = (pred_lengths[n]+1) if ((pred_lengths is not None) and (pred_lengths[n] < seq.shape[0])) else seq.shape[0]
            for i in range(seq_len):
                token = seq[i].item()
                if token != prev_token and token != blank_id:
                    collapsed_seq.append(token)
                prev_token = token
            
            decoded_preds.append(collapsed_seq)

        return decoded_preds
    


if __name__ == "__main__":
    
    #test the model:

    original_total_classes = 66
    blank_class = original_total_classes
    total_classes = original_total_classes + 1
    #test the layer size calculator:
    batch_size = 5
    seq_len = 20
    output = torch.randn(batch_size, seq_len, total_classes)
    targets = torch.randint(0, original_total_classes, (batch_size, seq_len))
    target_lengths = torch.randint(int(seq_len/2), seq_len, (batch_size,))
    
    print("Batch outputs size:", output.shape)
    print("Batch targets size:", targets.shape)
    print("Batch target_lengths size:", target_lengths.shape)
    print("Target lengths:", target_lengths)

    preds = F.log_softmax(output, dim=2)
    
    

    metrics_fn = MetricsCalculator5(blank_class=blank_class, total_classes=total_classes, total_entries=batch_size*2, ph_seq_max=seq_len, max_spec_len=seq_len, mode="greedy", device='cpu')

    metrics_fn.add_batch(preds, targets, target_lengths)
    metrics_fn.add_batch(preds, targets, target_lengths)

    

    results = metrics_fn.compute_sequential_metrics()
    print(results)

    
    metrics_fn.clear_mem()

