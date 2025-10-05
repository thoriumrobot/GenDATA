// Source-based slice around line 100
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putBytes(ByteBuffer)>


  @Override
  @CanIgnoreReturnValue
  public final Hasher putBytes(byte[] bytes, int off, int len) {
    return putBytesInternal(ByteBuffer.wrap(bytes, off, len).order(ByteOrder.LITTLE_ENDIAN));
  }

  @Override
  @CanIgnoreReturnValue
  public final Hasher putBytes(ByteBuffer readBuffer) {
    ByteOrder order = readBuffer.order();
    try {
      readBuffer.order(ByteOrder.LITTLE_ENDIAN);
      return putBytesInternal(readBuffer);
    } finally {
      readBuffer.order(order);
    }
  }

  @CanIgnoreReturnValue
