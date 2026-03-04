from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def compress(self, frames):
        """
        Compress a list of frames.

        Args:
            frames (list): A list of frames to be compressed.

        Returns:
            list: A list of compressed frames.
        """
        pass

    def write_compressed_data(self, compressed_frames, output_dir, batch_index: int):
        """
        Write compressed frames for a single batch to disk.
        Subclasses must implement this to match their storage format.
        """
        raise NotImplementedError

    def read_compressed_data(self, input_dir, batch_index: int):
        """
        Read compressed frames for a single batch from disk.
        Subclasses must implement this to match their storage format.

        Returns:
            list: A list of compressed frames for that batch.
        """
        raise NotImplementedError

    def compressed_batch_size_bytes(self, input_dir, batch_index: int) -> int:
        """
        Return the total on-disk size in bytes for a single compressed batch.
        Subclasses must implement this to match their storage format.
        """
        raise NotImplementedError

    @abstractmethod
    def decompress(self, compressed_frames):
        """
        Decompress a list of compressed frames.

        Args:
            compressed_frames (list): A list of compressed frames to be decompressed.

        Returns:
            list: A list of decompressed frames.
        """
        return compressed_frames