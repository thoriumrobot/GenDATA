// Source-based slice around line 139
// Method: <com.google.common.hash.AbstractByteHasher: ByteBuffer scratch()>


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
  }
}
