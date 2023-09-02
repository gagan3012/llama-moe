import datasets
from glob import glob
import json
import pandas as pd
import os

_BASE_URL = "/lustre07/scratch/gagan30/arocr/meta-llama/moe/dataset"

_TRAIN_DATA = glob(os.path.join(_BASE_URL, "*/*.zip"))

_DATA_DETAILS = {os.path.splitext(os.path.basename(f))[
    0]: f for f in _TRAIN_DATA}

print(_DATA_DETAILS)

_TASKS = [os.path.splitext(os.path.basename(f))[0] for f in _TRAIN_DATA]


class MoEDataConfig(datasets.BuilderConfig):
    """BuilderConfig for InverseScaling sample."""

    def __init__(self, *args, **kwargs):
        """BuilderConfig for InverseScaling.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MoEDataConfig, self).__init__(**kwargs)


class MoEData(datasets.GeneratorBasedBuilder):
    """MoE dataset."""

    BUILDER_CONFIG_CLASS = MoEDataConfig

    BUILDER_CONFIGS = []

    BUILDER_CONFIGS.extend(
        [
            MoEDataConfig(
                name=tasks,
                version=datasets.Version("1.0.0"),
                description="MoE dataset",
            )
            for tasks in _TASKS
        ]
    )

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "role_1": datasets.Value("string"),
                    "topic": datasets.Value("string"),
                    "sub_topic": datasets.Value("string"),
                    "message_1": datasets.Value("string"),
                    "message_2": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        print(_DATA_DETAILS)

        print(_DATA_DETAILS[self.config.name])
        downloaded_files = dl_manager.download(_DATA_DETAILS[self.config.name])

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={
                                        "filepath": downloaded_files})
        ]

    def _generate_examples(self, filepath):
        key = 0
        print(_DATA_DETAILS)
        for files in filepath:
            print(files)
            with open(filepath, encoding="utf-8") as f:
                for id_, row in enumerate(f):
                    row = json.loads(row)
                    yield key, {
                        "role_1": row["role_1"],
                        "topic": row["topic"],
                        "sub_topic": row["sub_topic"],
                        "message_1": row["message_1"],
                        "message_2": row["message_2"],
                    }
                    key += 1
