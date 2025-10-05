// Source-based slice around line 86
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putBytes(byte[])>

  @Override
  @CanIgnoreReturnValue
  public Hasher putByte(byte b) {
    update(b);
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes) {
    checkNotNull(bytes);
    update(bytes);
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes, int off, int len) {
    checkPositionIndexes(off, off + len, bytes.length);
    update(bytes, off, len);
