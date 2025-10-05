// Source-based slice around line 164
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putChar(char)>

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
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public final Hasher putInt(int i) {
    buffer.putInt(i);
    munchIfFull();
