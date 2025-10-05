// Source-based slice around line 156
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putShort(short)>

  @CanIgnoreReturnValue
  public final Hasher putByte(byte b) {
    buffer.put(b);
    munchIfFull();
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public final Hasher putShort(short s) {
    buffer.putShort(s);
    munchIfFull();
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public final Hasher putChar(char c) {
    buffer.putChar(c);
    munchIfFull();
