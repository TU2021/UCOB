# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess placeholder parquet for verl-agent (AlfWorld / WebShop / etc.).

We do NOT use row content from geometry3k at training time; only batch size matters.
See: https://github.com/langfengQ/verl-agent?tab=readme-ov-file#2-data-preparation
"""

import os
import argparse

import datasets

from verl.utils.hdfs_io import copy, makedirs

DEFAULT_GEOMETRY3K_LOCAL = "./xiaoyingzhang/datasets/hiyouga/geometry3k"


def _resolve_local_geometry3k(explicit_dir: str | None) -> str | None:
    candidates = [
        explicit_dir,
        os.environ.get("GEOMETRY3K_DIR"),
        DEFAULT_GEOMETRY3K_LOCAL,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


def _build_placeholder_split(split: str, size: int, mode: str):
    rows = []
    for idx in range(size):
        row = {
            "data_source": mode,
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": idx},
            "answer": "",
        }
        if mode == "visual":
            row["images"] = []
        rows.append(row)
    return datasets.Dataset.from_list(rows)


def _load_geometry3k_from_local(local_path: str, mode: str, train_data_size: int, val_data_size: int):
    print(f"loading local geometry3k from {local_path}")
    dataset = datasets.load_dataset(local_path)

    instruction_following = {"visual": "<image>", "text": ""}

    def make_map_fn(split):
        def process_fn(example, idx):
            example.pop("problem")
            prompt = instruction_following[mode]
            images = example.pop("images")
            if mode == "visual":
                data = {
                    "data_source": mode,
                    "prompt": [{"role": "user", "content": prompt}],
                    "images": images,
                    "ability": "agent",
                    "extra_info": {"split": split, "index": idx},
                }
            else:
                data = {
                    "data_source": mode,
                    "prompt": [{"role": "user", "content": prompt}],
                    "ability": "agent",
                    "extra_info": {"split": split, "index": idx},
                }
            return data

        return process_fn

    train_dataset = dataset["train"].select(range(train_data_size))
    test_dataset = dataset["test"].select(range(val_data_size))
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="visual", choices=["visual", "text"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_data_size", default=256, type=int)
    parser.add_argument("--val_data_size", default=256, type=int)
    parser.add_argument(
        "--geometry3k_dir",
        default=None,
        help="Local path to hiyouga/geometry3k (disk or HF cache). Never downloads from huggingface.co.",
    )

    args = parser.parse_args()
    print(f"processing data for mode: {args.mode}")
    args.local_dir = os.path.expanduser(os.path.join(args.local_dir, args.mode))
    os.makedirs(args.local_dir, exist_ok=True)

    train_path = os.path.join(args.local_dir, "train.parquet")
    test_path = os.path.join(args.local_dir, "test.parquet")

    local_geometry3k = _resolve_local_geometry3k(args.geometry3k_dir)
    if local_geometry3k is not None:
        train_dataset, test_dataset = _load_geometry3k_from_local(
            local_geometry3k, args.mode, args.train_data_size, args.val_data_size
        )
    else:
        print(
            f"local geometry3k not found (checked GEOMETRY3K_DIR / {DEFAULT_GEOMETRY3K_LOCAL}); "
            f"generating {args.train_data_size} train + {args.val_data_size} val placeholder rows"
        )
        train_dataset = _build_placeholder_split("train", args.train_data_size, args.mode)
        test_dataset = _build_placeholder_split("test", args.val_data_size, args.mode)

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"wrote {train_path} ({len(train_dataset)} rows), {test_path} ({len(test_dataset)} rows)")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
