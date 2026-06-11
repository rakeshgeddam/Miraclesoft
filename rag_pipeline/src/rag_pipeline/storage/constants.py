"""
Local vector storage with binary format and memory-mapped access.

Implements the HRID format from the RAG skill specification.
"""

import os
import json
import struct
import mmap
from dataclasses import dataclass
from typing import List, Optional, Iterator, Any
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Binary format constants
MAGIC_HRID = b"HRID"
MAGIC_IVF = b"IVF1"
MAGIC_HNSW = b"HNS1"
MAGIC_PQ = b"PQ01"

VERSION = 1
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1

FLAG_NORMALIZED = 1 << 0
FLAG_MATRYOSHKA = 1 << 1

HEADER_SIZE = 64