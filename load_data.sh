#!/usr/bin/env bash
set -euo pipefail

URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
ARCHIVE="CUB_200_2011.tgz"

# download
curl -L "$URL" -o "$ARCHIVE"

# unpack
tar -xzf "$ARCHIVE"
