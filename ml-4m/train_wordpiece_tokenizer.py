# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import hydra
from omegaconf import DictConfig
from fourm.utils.tokenizer import train_unified_wordpiece_tokenizer
from fourm.utils.tokenizer import generate_sentinel_tokens, generate_coord_tokens, generate_object_class_tokens


@hydra.main(config_path="../conf", config_name="config_wordpiece_tokenizer", version_base=None)
def train_tokenizer(cfg: DictConfig):

    files = cfg.text_files.split("--")
    # Get special tokens
    sentinel_tokens = generate_sentinel_tokens(num=cfg.num_sentinels)
    coord_tokens = generate_coord_tokens(bins=cfg.coord_bins)
    if cfg.object_classes == 'none':
        object_class_tokens = None
    else:
        object_class_tokens = generate_object_class_tokens(cfg.object_classes)

    print(f"Training tokenizer on files: {files}")

    # Train tokenizer
    tokenizer = train_unified_wordpiece_tokenizer(
        files=files,
        vocab_size=cfg.vocab_size,
        sentinel_tokens=sentinel_tokens,
        coord_tokens=coord_tokens,
        object_class_tokens=object_class_tokens,
        lowercase=cfg.lowercase,
    )

    # Create directory of target file if it doesn't exist
    os.makedirs(os.path.dirname(cfg.save_file), exist_ok=True)
    tokenizer.save(path=cfg.save_file)

    print(f"Tokenizer saved to: {cfg.save_file}!")


if __name__ == "__main__":
    train_tokenizer()
