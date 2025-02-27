from typing import List
import re
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser
from difflib import SequenceMatcher

class CorruptionRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["original"])

        def similarity_reward(completions, answer, **kwargs) -> List[float]:
            rewards = []
            for comp, orig in zip(completions, answer):
                comp_content = comp[-1]['content'].strip()

                # 1. Check proper closure
                if not comp_content.endswith('</original>'):
                    rewards.append(0.0)
                    continue

                # 2. Extract content (exclude </original>)
                restored = comp_content[:-len('</original>')].strip()

                # 3. Check for prohibited tags
                if any(tag in restored.lower() for tag in ['<original>', '<objective>']):
                    rewards.append(0.0)
                    continue

                # 4. Compute similarity
                matcher = SequenceMatcher(None, orig, restored)
                rewards.append(matcher.ratio())
            return rewards

        def length_reward(completions, answer, **kwargs) -> List[float]:
            rewards = []
            for comp, orig in zip(completions, answer):
                comp_content = comp[-1]['content'].strip()

                if not comp_content.endswith('</original>'):
                    rewards.append(0.0)
                    continue

                restored = comp_content[:-len('</original>')].strip()
                orig_len = len(orig)
                comp_len = len(restored)
                threshold = orig_len * 0.1

                if abs(comp_len - orig_len) <= threshold:
                    rewards.append(0.1)
                else:
                    rewards.append(0.0)
            return rewards

        self.reward_funcs = [similarity_reward, length_reward]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs