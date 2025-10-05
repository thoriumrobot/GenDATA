// Source-based slice around line 102
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putBytes(ByteBuffer)>

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
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putShort(short s) {
    ByteBuffer scratch = scratch();
    scratch.putShort(s);
    return update(scratch, Shorts.BYTES);
