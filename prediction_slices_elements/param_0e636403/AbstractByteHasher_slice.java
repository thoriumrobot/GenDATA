// Source-based slice around line 109
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putShort(short)>

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
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putInt(int i) {
    ByteBuffer scratch = scratch();
    scratch.putInt(i);
