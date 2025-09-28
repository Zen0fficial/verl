import unittest
from unittest.mock import MagicMock

import torch

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class TestKeypointReward(unittest.TestCase):
    def test_compute_keypoint_reward_scores(self):
        # Create trainer instance without invoking __init__
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)

        # Mock tokenizer to produce single-token IDs for any keypoint string
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.encode = lambda s, **kwargs: [1]  # every keypoint -> [1]

        # Mock actor_rollout_wg.compute_log_prob to control log-probabilities
        # We will have response_len = 3 and bs = 1.
        # There will be one keypoint of length 1, so variants == number of valid starts == 3.
        # Arrange that for the 3 variants, the window probs at their respective start positions are 0.1, 0.2, 0.3.
        # That implies:
        #   mod_log_probs[0,0] = log(0.1)
        #   mod_log_probs[1,1] = log(0.2)
        #   mod_log_probs[2,2] = log(0.3)
        # Other positions can be set to log(1.0) to be neutral for k_len=1 windows.
        class _MockWG:
            def compute_log_prob(self, dp: DataProto) -> DataProto:
                # dp contains variants stacked in dim=0; response_len should be 3 here
                num_variants = dp.batch["responses"].shape[0]
                response_len = dp.batch["responses"].shape[1]
                assert response_len == 3
                assert num_variants == 3
                log_probs = torch.log(torch.tensor(
                    [
                        [0.1, 1.0, 1.0],
                        [1.0, 0.2, 1.0],
                        [1.0, 1.0, 0.3],
                    ],
                    dtype=torch.float32,
                ))
                return DataProto.from_dict(tensors={"old_log_probs": log_probs})

        trainer.actor_rollout_wg = _MockWG()

        # Build a minimal DataProto expected by _compute_keypoint_reward_scores
        bs = 1
        prompt_len = 2
        response_len = 3
        seq_len = prompt_len + response_len

        input_ids = torch.zeros((bs, seq_len), dtype=torch.long)
        attention_mask = torch.ones((bs, seq_len), dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(bs, -1)
        responses = torch.zeros((bs, response_len), dtype=torch.long)
        response_mask = torch.ones((bs, response_len), dtype=torch.long)

        # Put keypoints under extra_info (new convention)
        extra_info = [
            {"keypoints": ["Relativity theory", "Albert Einstein"]},
        ]

        batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            non_tensors={
                "extra_info": extra_info,  # will be converted to np.ndarray(dtype=object) with shape (bs,)
            },
        )

        # Run keypoint reward computation
        scores, extra = trainer._compute_keypoint_reward_scores(batch)

        # Shape checks
        self.assertEqual(scores.shape, (bs, response_len))

        # The reward should be placed at the last masked token position (index 2)
        last_idx = 2

        # Expected existence probability: 1 - (1-0.2)*(1-0.5) = 0.6
        expected_seq_score = torch.log(torch.tensor(0.6, dtype=torch.float32))

        self.assertTrue(torch.allclose(scores[0, last_idx], expected_seq_score, atol=1e-5))
        # All other positions should be zero
        self.assertTrue(torch.all(scores[0, :last_idx] == 0))