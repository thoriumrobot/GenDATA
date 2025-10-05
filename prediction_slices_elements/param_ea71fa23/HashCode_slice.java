// Source-based slice around line 84
// Method: <com.google.common.hash.HashCode: int writeBytesTo(byte[],int,int)>

   * Copies bytes from this hash code into {@code dest}.
   *
   * @param dest the byte array into which the hash code will be written
   * @param offset the start offset in the data
   * @param maxLength the maximum number of bytes to write
   * @return the number of bytes written to {@code dest}
   * @throws IndexOutOfBoundsException if there is not enough room in {@code dest}
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
