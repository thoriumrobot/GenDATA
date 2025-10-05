// Source-based slice around line 125
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putLong(long)>

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
    return update(scratch, Longs.BYTES);
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putChar(char c) {
    ByteBuffer scratch = scratch();
    scratch.putChar(c);
