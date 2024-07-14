# paper-parser

## Usage

### Install
```
pip install https://github.com/akitenkrad/paper-parser.git
```

### Extract texts from a paper
```python
from paper_parser import parse_from_file, parse_from_url
texts = parse_from_file("<PATH TO A PAPER PDF>")
# texts = parse_from_url("<URL OF THE PAPER PDF>")
```