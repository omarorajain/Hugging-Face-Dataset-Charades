# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

# Lint as: python3
"""Charades is dataset composed of 9848 videos of daily indoors activities collected through Amazon Mechanical Turk"""
import csv
import os
import tarfile
import zipfile

import datasets
import requests
from tqdm import tqdm

from .classes import CHARADES_CLASSES

_CITATION = """
@article{sigurdsson2016hollywood,
    author = {Gunnar A. Sigurdsson and G{\"u}l Varol and Xiaolong Wang and Ivan Laptev and Ali Farhadi and Abhinav Gupta},
    title = {Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding},
    journal = {ArXiv e-prints},
    eprint = {1604.01753},
    year = {2016},
    url = {http://arxiv.org/abs/1604.01753},
}
"""

_DESCRIPTION = """\
Charades is dataset composed of 9848 videos of daily indoors activities collected through Amazon Mechanical Turk. 267 different users were presented with a sentence, that includes objects and actions from a fixed vocabulary, and they recorded a video acting out the sentence (like in a game of Charades). The dataset contains 66,500 temporal annotations for 157 action classes, 41,104 labels for 46 object classes, and 27,847 textual descriptions of the videos.
"""


_ANNOTATIONS_URL = (
    "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
)
_VIDEOS_URL = {
    "default": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip",
    "480p": "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip",
}


def downloadWithProgress(url, localFilename):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    totalSize = int(response.headers.get("content-length", 0))

    with tqdm(
        desc=f"Downloading {os.path.basename(localFilename)}",
        total=totalSize,
        unit="MB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        with open(localFilename, "wb") as file:
            for data in response.iter_content(8192):
                bar.update(len(data))
                file.write(data)

    if totalSize != 0 and bar.n != totalSize:
        raise RuntimeError("Could not download file")
    
    return localFilename


def extractWithProgress(localFilename, extractDir):
    """Extract a file with a progress bar."""
    print(f"Extracting {localFilename}...")

    if localFilename.endswith(".zip"):
        with zipfile.ZipFile(localFilename, "r") as zipRef:
            fileList = zipRef.namelist()
            with tqdm(
                desc=f"Extracting {os.path.basename(localFilename)}",
                total=len(fileList),
                unit="file",
            ) as bar:
                for file in fileList:
                    zipRef.extract(file, extractDir)
                    bar.update(1)

    elif localFilename.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(localFilename, "r:*") as tarRef:
            members = tarRef.getmembers()
            with tqdm(
                desc=f"Extracting {os.path.basename(localFilename)}",
                total=len(members),
                unit="file",
            ) as bar:
                for member in members:
                    tarRef.extract(member, extractDir)
                    bar.update(1)

    else:
        print(f"Unknown file format for {localFilename}, skipping extraction.")

    print(f"Extraction complete for {localFilename}.")
    return extractDir


class Charades(datasets.GeneratorBasedBuilder):
    """Charades is dataset composed of 9848 videos of daily indoors activities collected through Amazon Mechanical Turk"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default"),
        datasets.BuilderConfig(name="480p"),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "video_id": datasets.Value("string"),
                    "video": datasets.Value("string"),
                    "subject": datasets.Value("string"),
                    "scene": datasets.Value("string"),
                    "quality": datasets.Value("int32"),
                    "relevance": datasets.Value("int32"),
                    "verified": datasets.Value("string"),
                    "script": datasets.Value("string"),
                    "objects": datasets.features.Sequence(datasets.Value("string")),
                    "descriptions": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            num_classes=len(CHARADES_CLASSES),
                            names=list(CHARADES_CLASSES.values()),
                        )
                    ),
                    "action_timings": datasets.Sequence(
                        datasets.Sequence(datasets.Value("float32"))
                    ),
                    "length": datasets.Value("float32"),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        DOWNLOAD_DIR = "./downloads"
        EXTRACT_DIR = "./extracted"

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(EXTRACT_DIR, exist_ok=True)

        def download_and_extract(url, downloadDir, extractDir):
            """Download and extract a file with progress bars."""
            # Download file
            localFilename = os.path.join(downloadDir, os.path.basename(url))
            downloadWithProgress(url, localFilename)

            # Extract file
            extractedPath = extractWithProgress(localFilename, extractDir)
            return extractedPath

        # annotations_path = dl_manager.download_and_extract(_ANNOTATIONS_URL)
        # archive = os.path.join(dl_manager.download_and_extract(_VIDEOS_URL[self.config.name]), "Charades_v1")

        annotations_path = download_and_extract(
            _ANNOTATIONS_URL, DOWNLOAD_DIR, EXTRACT_DIR
        )
        archive = os.path.join(
            download_and_extract(
                _VIDEOS_URL[self.config.name], DOWNLOAD_DIR, EXTRACT_DIR
            ),
            "Charades_v1",
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": os.path.join(
                        annotations_path, "Charades", "Charades_v1_train.csv"
                    ),
                    "video_folder": archive,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": os.path.join(
                        annotations_path, "Charades", "Charades_v1_test.csv"
                    ),
                    "video_folder": archive,
                },
            ),
        ]

    def _generate_examples(self, annotation_file, video_folder):
        """This function returns the examples."""
        with open(annotation_file, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            idx = 0
            for row in reader:
                path = os.path.join(video_folder, row["id"] + ".mp4")
                labels = []
                action_timings = []
                for class_label in row["actions"].split(";"):
                    # Skip empty action labels
                    if len(class_label) != 0:
                        # format is like: "c123 11.0 13.0"
                        labels.append(CHARADES_CLASSES[class_label.split(" ")[0]])
                        timings = list(map(float, class_label.split(" ")[1:]))
                        action_timings.append(timings)

                yield idx, {
                    "video_id": row["id"],
                    "video": path,
                    "subject": row["subject"],
                    "scene": row["scene"],
                    "quality": (
                        int(row["quality"]) if len(row["quality"]) != 0 else -100
                    ),
                    "relevance": (
                        int(row["relevance"]) if len(row["relevance"]) != 0 else -100
                    ),
                    "verified": row["verified"],
                    "script": row["script"],
                    "objects": row["objects"].split(";"),
                    "descriptions": row["descriptions"].split(";"),
                    "labels": labels,
                    "action_timings": action_timings,
                    "length": row["length"],
                }

                idx += 1