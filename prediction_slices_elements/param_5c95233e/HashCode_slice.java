// Source-based slice around line 91
// Method: <com.google.common.hash.HashCode: void writeBytesToImpl(byte[],int,int)>

   */
  @CanIgnoreReturnValue
  public int writeBytesTo(byte[] dest, int offset, int maxLength) {
    maxLength = min(maxLength, bits() / 8);
    Preconditions.checkPositionIndexes(offset, offset + maxLength, dest.length);
    writeBytesToImpl(dest, offset, maxLength);
    return maxLength;
  }

  abstract void writeBytesToImpl(byte[] dest, int offset, int maxLength);

  /**
   * Returns a mutable view of the underlying bytes for the given {@code HashCode} if it is a
   * byte-based hashcode. Otherwise it returns {@link HashCode#asBytes}. Do <i>not</i> mutate this
   * array or else you will break the immutability contract of {@code HashCode}.
   */
  byte[] getBytesInternal() {
    return asBytes();
  }

