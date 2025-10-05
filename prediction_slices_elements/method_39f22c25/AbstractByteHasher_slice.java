// Source-based slice around line 94
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putBytes(byte[],int,int)>

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
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putBytes(ByteBuffer bytes) {
    update(bytes);
    return this;
