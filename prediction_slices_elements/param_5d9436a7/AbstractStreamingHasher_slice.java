// Source-based slice around line 172
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putInt(int)>

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
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public final Hasher putLong(long l) {
    buffer.putLong(l);
    munchIfFull();
