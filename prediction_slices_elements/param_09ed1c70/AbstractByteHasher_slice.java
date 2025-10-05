// Source-based slice around line 133
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putChar(char)>

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
    return update(scratch, Chars.BYTES);
  }

  private ByteBuffer scratch() {
    if (scratch == null) {
      scratch = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    }
    return scratch;
