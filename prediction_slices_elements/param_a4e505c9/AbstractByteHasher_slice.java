// Source-based slice around line 117
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putInt(int)>

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
    return update(scratch, Ints.BYTES);
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putLong(long l) {
    ByteBuffer scratch = scratch();
    scratch.putLong(l);
