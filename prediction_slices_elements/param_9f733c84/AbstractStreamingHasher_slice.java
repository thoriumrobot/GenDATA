// Source-based slice around line 81
// Method: <com.google.common.hash.AbstractStreamingHasher: void processRemaining(ByteBuffer)>

  /** Processes the available bytes of the buffer (at most {@code chunk} bytes). */
  protected abstract void process(ByteBuffer bb);

  /**
   * This is invoked for the last bytes of the input, which are not enough to fill a whole chunk.
   * The passed {@code ByteBuffer} is guaranteed to be non-empty.
   *
   * <p>This implementation simply pads with zeros and delegates to {@link #process(ByteBuffer)}.
   */
  protected void processRemaining(ByteBuffer bb) {
    Java8Compatibility.position(bb, bb.limit()); // move at the end
    Java8Compatibility.limit(bb, chunkSize + 7); // get ready to pad with longs
    while (bb.position() < chunkSize) {
      bb.putLong(0);
    }
    Java8Compatibility.limit(bb, chunkSize);
    Java8Compatibility.flip(bb);
    process(bb);
  }

