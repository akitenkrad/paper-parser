from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import wordnet as wn

if not (Path(nltk.downloader.Downloader().default_download_dir()) / "corpora/wordnet.zip").exists():
    try:
        nltk.download("wordnet")
        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("averaged_perceptron_tagger")
    except Exception as e:
        print(f"Failed to download wordnet corpus: {e}")
        print("trying to download wordnet with alternative way.")
        subprocess.run(
            "python -m nltk.downloader wordnet punkt punkt_tab averaged_perceptron_tagger", shell=True, check=True
        )


class ElementType(Enum):
    FigureCaption = "FigureCaption"
    Footer = "Footer"
    Header = "Header"
    Image = "Image"
    ListItem = "ListItem"
    NarrativeText = "NarrativeText"
    Table = "Table"
    Title = "Title"
    UncategorizedText = "UncategorizedText"

    @classmethod
    def parse(cls, value: str) -> ElementType:
        for e in ElementType:
            if e.value == value:
                return e
        return ElementType.UncategorizedText


class HeaderType(Enum):
    FirstHeader = "FirstHeader"
    SecondHeader = "SecondHeader"
    ThirdHeader = "ThirdHeader"
    FourthHeader = "FourthHeader"
    FifthHeader = "FifthHeader"
    AppendixHeader = "AppendixHeader"
    Unknown = "Unknown"


@dataclass
class Point(object):
    x: int
    y: int

    def pos(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass
class Coordinates(object):
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point

    def width(self) -> int:
        return self.top_right.x - self.top_left.x

    def height(self) -> int:
        return self.bottom_left.y - self.top_left.y

    def is_intercept(self, other: Coordinates) -> float:
        left = np.min([self.top_left.x, other.top_left.x])
        right = np.max([self.top_right.x, other.top_right.x])
        top = np.min([self.top_left.y, other.top_left.y])
        bottom = np.max([self.bottom_left.y, other.bottom_left.y])

        return right - left < self.width() + other.width() and bottom - top < self.height() + other.height()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Coordinates):
            raise NotImplementedError

        if self.top_right.x <= other.top_left.x:
            return True
        elif self.top_left.x >= other.top_right.x:
            return False
        else:
            return self.top_left.y < other.top_left.y


@dataclass
class Element(object):
    type: ElementType
    text: str
    page_number: int
    coordinates: Coordinates
    layout_width: int
    layout_height: int
    file_directory: str
    filename: str
    filetype: str
    languages: list[str] = field(default_factory=list)

    def __post_init__(self):
        new_tokens = []
        tokens = self.text.split(" ")
        skip = False
        for idx in range(len(tokens) - 1):
            prev_token = tokens[idx]
            next_token = tokens[idx + 1]

            if next_token.endswith("."):
                next_token = next_token[:-1]

            if skip:
                skip = False
                continue
            if (prev_token + next_token).lower() in ["end-to-end", "state-of-the-art"]:
                new_tokens.append(prev_token + next_token)
                skip = True
            elif prev_token.endswith("-"):
                if wn.morphy(prev_token[:-1]) and wn.morphy(next_token):
                    new_tokens.append(prev_token + next_token)
                    skip = True
                else:
                    new_tokens.append(prev_token[:-1] + next_token)
                    skip = True
            else:
                new_tokens.append(prev_token)

        new_tokens.append(tokens[-1])
        self.text = " ".join(new_tokens)

    @classmethod
    def from_dict(cls, data: dict) -> Element:
        meta = data["metadata"]
        return cls(
            type=ElementType.parse(data["type"]),
            text=data["text"],
            page_number=meta["page_number"],
            coordinates=Coordinates(
                top_left=Point(*[int(v) for v in meta["coordinates"]["points"][0]]),
                top_right=Point(*[int(v) for v in meta["coordinates"]["points"][3]]),
                bottom_left=Point(*[int(v) for v in meta["coordinates"]["points"][1]]),
                bottom_right=Point(*[int(v) for v in meta["coordinates"]["points"][2]]),
            ),
            layout_width=meta["coordinates"]["layout_width"],
            layout_height=meta["coordinates"]["layout_height"],
            languages=meta.get("languages", ""),
            file_directory=meta.get("file_directory", ""),
            filename=meta.get("filename", ""),
            filetype=meta.get("filetype", ""),
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Element):
            raise NotImplementedError
        return self.coordinates < other.coordinates
