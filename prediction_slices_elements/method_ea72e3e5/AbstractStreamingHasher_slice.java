// Source-based slice around line 212
// Method: <com.google.common.hash.AbstractStreamingHasher: void munch()>


  // Process pent-up data in chunks
  private void munchIfFull() {
    if (buffer.remaining() < 8) {
      // buffer is full; not enough room for a primitive. We have at least one full chunk.
      munch();
    }
  }

  private void munch() {
    Java8Compatibility.flip(buffer);
    while (buffer.remaining() >= chunkSize) {
      // we could limit the buffer to ensure process() does not read more than
      // chunkSize number of bytes, but we trust the implementations
      process(buffer);
    }
    buffer.compact(); // preserve any remaining data that do not make a full chunk
  }
}
