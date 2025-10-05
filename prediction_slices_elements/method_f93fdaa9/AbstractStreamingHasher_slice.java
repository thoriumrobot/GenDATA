// Source-based slice around line 180
// Method: <com.google.common.hash.AbstractStreamingHasher: Hasher putLong(long)>

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
    return this;
  }

  @Override
  public final HashCode hash() {
    munch();
    Java8Compatibility.flip(buffer);
    if (buffer.remaining() > 0) {
