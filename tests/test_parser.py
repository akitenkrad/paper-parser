from research_paper_parser.parser import parse_from_url


def test_parse_from_url_1():

    url = "https://arxiv.org/pdf/2106.01484.pdf"
    elements = parse_from_url(url)
    assert len(elements) > 0


def test_parse_from_url_2():

    url = "https://arxiv.org/pdf/1810.04805"
    elements = parse_from_url(url)
    assert len(elements) > 0
