// Source-based slice around line 111
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putBytesInternal(ByteBuffer)>

    try {
      readBuffer.order(ByteOrder.LITTLE_ENDIAN);
      return putBytesInternal(readBuffer);
    } finally {
      readBuffer.order(order);
    }
  }

  @CanIgnoreReturnValue
  private Hasher putBytesInternal(ByteBuffer readBuffer) {
    // If we have room for all of it, this is easy
    if (readBuffer.remaining() <= buffer.remaining()) {
      buffer.put(readBuffer);
      munchIfFull();
      return this;
    }

    // First add just enough to fill buffer size, and munch that
    int bytesToCopy = bufferSize - buffer.position();
    for (int i = 0; i < bytesToCopy; i++) {
