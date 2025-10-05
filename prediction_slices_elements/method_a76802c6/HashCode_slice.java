// Source-based slice around line 114
// Method: <com.google.common.hash.HashCode: HashCode fromInt(int)>

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

  private static final class IntHashCode extends HashCode implements Serializable {
    final int hash;

    IntHashCode(int hash) {
      this.hash = hash;
    }

