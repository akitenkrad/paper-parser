#!/bin/bash

set -eu

apt update -y
apt install -y libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc

