import re
import urllib
from logging import Logger
from typing import Optional

import numpy as np
from unstructured.partition.pdf import partition_pdf

from paper_parser.data import Coordinates, Element, ElementType, HeaderType, Point


def is_reference_section(element: Element) -> bool:
    ptn = re.compile(r"references?$", re.IGNORECASE)
    return element.type == ElementType.Title and ptn.search(element.text.strip()) is not None and len(element.text) < 15


def get_header_type(element: Element) -> HeaderType:
    if re.search(r"^(\d\.?)\s", element.text):
        return HeaderType.FirstHeader
    elif re.search(r"^(\d\.\d\.?)\s", element.text):
        return HeaderType.SecondHeader
    elif re.search(r"^(\d\.\d\.\d\.?)\s", element.text):
        return HeaderType.ThirdHeader
    elif re.search(r"^(\d\.\d\.\d\.\d\.?)\s", element.text):
        return HeaderType.FourthHeader
    elif re.search(r"^(\d\.\d\.\d\.\d\.\d\.?)\s", element.text):
        return HeaderType.FifthHeader
    elif element.text.lower().strip().startswith("appendix"):
        return HeaderType.AppendixHeader
    else:
        return HeaderType.Unknown


def is_title(element: Element, elements: list[Element]) -> bool:
    height_list = [e.coordinates.height() for e in elements if e.type == ElementType.Title]
    l = np.mean(height_list) - np.std(height_list) * 3
    h = np.mean(height_list) + np.std(height_list) * 3
    return l < element.coordinates.height() < h


def is_in_text_area(element: Element, text_area: Coordinates, th: float = 0.7):
    left = np.max([element.coordinates.top_left.x, text_area.top_left.x])
    right = np.min([element.coordinates.top_right.x, text_area.top_right.x])
    top = np.max([element.coordinates.top_left.y, text_area.top_left.y])
    bottom = np.min([element.coordinates.bottom_left.y, text_area.bottom_left.y])

    width = np.max([0, right - left])
    height = np.max([0, bottom - top])

    area = width * height
    element_area = element.coordinates.width() * element.coordinates.height()

    return area / element_area > th


def is_part_of_table(element: Element, elements: list[Element]) -> bool:
    if element.type == ElementType.Table:
        return True

    table_elems = [e for e in elements if e.type == ElementType.Table and e.page_number == element.page_number]
    for table_elem in table_elems:
        if element.page_number != table_elem.page_number:
            continue

        if element.coordinates.is_intercept(table_elem.coordinates):
            return True

    return False


def is_figure_caption(element: Element, elements: list[Element], th: int = 50) -> bool:
    if element.type == ElementType.FigureCaption:
        return True

    image_elems = [e for e in elements if e.type == ElementType.Image and e.page_number == element.page_number]
    if len(image_elems) == 0:
        return False

    for image_elem in image_elems:
        if element.page_number != image_elem.page_number:
            continue

        if element.coordinates.is_intercept(image_elem.coordinates) and element.text.lower().startswith("fig"):
            return True

        y_diff = element.coordinates.top_left.y - image_elem.coordinates.bottom_left.y
        if 0 < y_diff < th and element.text.lower().startswith("fig"):
            return True

    return False


def is_table_caption(element: Element, elements: list[Element], th: int = 50) -> bool:
    if element.type == ElementType.FigureCaption:
        return True

    table_elems = [e for e in elements if e.type == ElementType.Table and e.page_number == element.page_number]
    if len(table_elems) == 0:
        return False

    for table_elem in table_elems:
        if element.page_number != table_elem.page_number:
            continue

        if element.coordinates.is_intercept(table_elem.coordinates) and element.text.lower().startswith("table"):
            return True

        y_diff = table_elem.coordinates.top_left.y - element.coordinates.bottom_left.y
        if 0 < y_diff < th and element.text.lower().startswith("table"):
            return True

    return False


def get_text_area(elements: list[Element]) -> Coordinates:
    target_types = [
        ElementType.NarrativeText,
        ElementType.ListItem,
        ElementType.Image,
        ElementType.Table,
        ElementType.FigureCaption,
    ]

    page_count = max([element.page_number for element in elements])

    top = [list() for _ in range(page_count)]
    for page in range(1, page_count + 1):
        top[page - 1] = [
            element.coordinates.top_left.y
            for element in elements
            if element.page_number == page and element.coordinates.top_left.y > 0 and element.type in target_types
        ]
    top = [np.min(values) if len(values) > 0 else 0 for values in top]

    left = [list() for _ in range(page_count)]
    for page in range(1, page_count + 1):
        left[page - 1] = [
            element.coordinates.top_left.x
            for element in elements
            if element.page_number == page and element.coordinates.top_left.x > 0 and element.type in target_types
        ]
    left = [np.min(values) if len(values) > 0 else 0 for values in left]

    right = [list() for _ in range(page_count)]
    for page in range(1, page_count + 1):
        right[page - 1] = [
            element.coordinates.top_right.x
            for element in elements
            if element.page_number == page and element.type in target_types
        ]
    right_max = np.max([np.max(v) for v in right if len(v) > 0])
    right = [np.max(values) if len(values) > 0 else right_max for values in right]

    bottom = [list() for _ in range(page_count)]
    for page in range(1, page_count + 1):
        bottom[page - 1] = [
            element.coordinates.bottom_left.y
            for element in elements
            if element.page_number == page and element.type in target_types
        ]
    bottom_max = np.max([np.max(v) for v in bottom if len(v) > 0])
    bottom = [np.max(values) if len(values) > 0 else bottom_max for values in bottom]

    t, l, r, b = np.median(top), np.median(left), np.median(right), np.median(bottom)
    return Coordinates(Point(l, t), Point(r, t), Point(l, b), Point(r, b))


def sort_elements(elements: list[Element]) -> list[Element]:
    elements_by_page = [list() for _ in range(max([element.page_number for element in elements]))]
    for element in elements:
        elements_by_page[element.page_number - 1].append(element)

    results = []
    for elems in elements_by_page:
        results.append(sorted(elems))

    return [elem for elems in results for elem in elems]


def adjust_width(elements: list[Element], text_area: Coordinates) -> list[Element]:
    width = text_area.width() / 2.2
    avg_width = np.mean([e.coordinates.width() for e in elements if e.type == ElementType.NarrativeText])
    if avg_width < text_area.width() / 1.5:
        # Two column paper
        for e in elements:
            if e.coordinates.top_right.x < text_area.top_left.x + text_area.width() / 2:
                e.coordinates.top_left.x = text_area.top_left.x + 10
                e.coordinates.bottom_left.x = text_area.bottom_left.x + 10
                e.coordinates.top_right.x = text_area.top_left.x + width - 10
                e.coordinates.bottom_right.x = text_area.top_left.x + width - 10
            else:
                e.coordinates.top_left.x = text_area.top_right.x - width - 10
                e.coordinates.bottom_left.x = text_area.bottom_right.x - width - 10
                e.coordinates.top_right.x = text_area.top_right.x - 10
                e.coordinates.bottom_right.x = text_area.bottom_right.x - 10
    else:
        # Single column paper
        for e in elements:
            e.coordinates.top_left.x = text_area.top_left.x + 10
            e.coordinates.bottom_left.x = text_area.bottom_left.x + 10
            e.coordinates.top_right.x = text_area.top_right.x - 10
            e.coordinates.bottom_right.x = text_area.bottom_right.x - 10


def parse_from_url(url: str, logger: Optional[Logger] = None) -> dict:
    data = urllib.request.urlopen(url).read()
    with open("/tmp/paper.pdf", mode="wb") as f:
        f.write(data)
    partitions = partition_pdf(filename="/tmp/paper.pdf", strategy="hi_res")

    if logger:
        logger.info(f"Number of partitions: {len(partitions)}")

    elements = [Element.from_dict(partition.to_dict()) for partition in partitions]
    text_area = get_text_area(elements)
    adjust_width(elements, text_area)
    elements = sort_elements(elements)

    if logger:
        logger.info("Number of pages: " + str(max([element.page_number for element in elements])))
        logger.info("Number of elements: " + str(len(elements)))

    text_types = [
        ElementType.Title,
        ElementType.NarrativeText,
        ElementType.ListItem,
    ]

    text_elements = []
    for element in elements:
        if is_reference_section(element):
            break
        if (
            element.type in text_types
            and is_in_text_area(element, text_area)
            and not is_figure_caption(element, elements)
            and not is_table_caption(element, elements)
            and not is_part_of_table(element, elements)
            and not (element.type == ElementType.Title and not is_title(element, elements))
        ):
            text_elements.append(element)

    if logger:
        logger.info("Number of text elements: " + str(len(text_elements)))

    current_section = "Abstract"
    texts = {current_section: ""}
    for element in text_elements:
        if element.type == ElementType.Title:

            if logger:
                logger.info(f"Processing: {element.text}")

            if "abstract" in element.text.lower():
                current_section = "Abstract"
                texts[current_section] = ""
                continue
            if "introduction" in element.text.lower():
                current_section = element.text
                texts[current_section] = ""
                continue
            header_type = get_header_type(element)
            if header_type in [HeaderType.FirstHeader, HeaderType.AppendixHeader]:
                current_section = element.text.strip()
                texts[current_section] = ""
                continue
        texts[current_section] += element.text.strip() + " "

    text_keys = list(texts.keys())
    for k in text_keys:
        v = texts[k]
        if len(v) == 0:
            del texts[k]
        else:
            texts[k] = v.strip()

    return texts


def parse_from_file(pdf_path: str, logger: Optional[Logger] = None) -> dict:
    partitions = partition_pdf(filename=pdf_path, strategy="hi_res")

    if logger:
        logger.info(f"Number of partitions: {len(partitions)}")

    elements = [Element.from_dict(partition.to_dict()) for partition in partitions]
    text_area = get_text_area(elements)
    adjust_width(elements, text_area)
    elements = sort_elements(elements)

    if logger:
        logger.info("Number of pages: " + str(max([element.page_number for element in elements])))
        logger.info("Number of elements: " + str(len(elements)))

    text_types = [
        ElementType.Title,
        ElementType.NarrativeText,
        ElementType.ListItem,
    ]

    text_elements = []
    for element in elements:
        if is_reference_section(element):
            break
        if (
            element.type in text_types
            and is_in_text_area(element, text_area)
            and not is_figure_caption(element, elements)
            and not is_table_caption(element, elements)
            and not is_part_of_table(element, elements)
            and not (element.type == ElementType.Title and not is_title(element, elements))
        ):
            text_elements.append(element)

    if logger:
        logger.info("Number of text elements: " + str(len(text_elements)))

    current_section = "Abstract"
    texts = {current_section: ""}
    for element in text_elements:
        if element.type == ElementType.Title:

            if logger:
                logger.info(f"Processing: {element.text}")

            if "abstract" in element.text.lower():
                current_section = "Abstract"
                texts[current_section] = ""
                continue
            if "introduction" in element.text.lower():
                current_section = element.text
                texts[current_section] = ""
                continue
            header_type = get_header_type(element)
            if header_type in [HeaderType.FirstHeader, HeaderType.AppendixHeader]:
                current_section = element.text.strip()
                texts[current_section] = ""
                continue
        texts[current_section] += element.text.strip() + " "

    text_keys = list(texts.keys())
    for k in text_keys:
        v = texts[k]
        if len(v) == 0:
            del texts[k]
        else:
            texts[k] = v.strip()

    return texts
