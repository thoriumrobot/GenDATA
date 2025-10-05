// Source-based slice around line 106
// Method: <com.google.common.hash.HashCode: boolean equalsSameBits(HashCode)>

   */
  byte[] getBytesInternal() {
    return asBytes();
  }

  /**
   * Returns whether this {@code HashCode} and that {@code HashCode} have the same value, given that
   * they have the same number of bits.
   */
  abstract boolean equalsSameBits(HashCode that);

  /**
   * Creates a 32-bit {@code HashCode} representation of the given int value. The underlying bytes
   * are interpreted in little endian order.
   *
   * @since 15.0 (since 12.0 in HashCodes)
   */
  public static HashCode fromInt(int hash) {
    return new IntHashCode(hash);
  }
