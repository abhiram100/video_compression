from compressor.compressors.base_compressor import BaseCompressor

class IdentityCompressor(BaseCompressor):
    def compress(self, frames):
        return frames

    def write_compressed_data(self, compressed_frames, output_dir, batch_index: int):
        pass  # No-op since there's no actual compression

    def read_compressed_data(self, input_dir, batch_index: int):
        pass  # No-op since there's no actual compression

    def decompress(self, compressed_frames):
        return compressed_frames