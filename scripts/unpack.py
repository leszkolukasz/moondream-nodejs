# Based on https://github.com/vikhyat/moondream

import argparse
import struct
import gzip
import os
from typing import BinaryIO, Tuple, Iterator, Union

MOON_MAGIC = b"MOON"
MOON_VERSION = 1


class MoonReader:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def _get_file_handle(self) -> Union[BinaryIO, gzip.GzipFile]:
        """Returns appropriate file handle based on extension"""
        if self.input_path.endswith(".gz"):
            return gzip.open(self.input_path, "rb")
        return open(self.input_path, "rb")

    def _validate_header(self, f: Union[BinaryIO, gzip.GzipFile]) -> None:
        """Validate magic bytes and version"""
        magic = f.read(4)
        if magic != MOON_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")

        version = struct.unpack("!B", f.read(1))[0]
        if version != MOON_VERSION:
            raise ValueError(f"Unsupported version: {version}")

    def read_files(self) -> Iterator[Tuple[str, bytes]]:
        """Read and yield (filename, content) pairs from the archive"""
        with self._get_file_handle() as f:
            self._validate_header(f)

            while True:
                # Try to read filename length
                filename_len_bytes = f.read(4)
                if not filename_len_bytes:
                    break  # End of file

                filename_len = struct.unpack("!I", filename_len_bytes)[0]

                # Read filename
                filename = f.read(filename_len).decode("utf-8")

                # Read content length and content
                content_len = struct.unpack("!Q", f.read(8))[0]
                content = f.read(content_len)

                yield filename, content


def unpack(input_path: str) -> Iterator[Tuple[str, bytes]]:
    """Unpack a .mf file"""
    reader = MoonReader(input_path)
    for filename, content in reader.read_files():
        yield filename, content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack a .mf file")
    parser.add_argument("input_path", type=str, help="Path to the .mf file")
    parser.add_argument("output_dir", type=str, help="Directory to save unpacked files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename, content in unpack(args.input_path):
        output_path = os.path.join(args.output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(content)

        print(f"Unpacked {filename} with size {len(content)} bytes")