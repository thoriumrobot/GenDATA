// Source-based slice around line 171
// Method: <com.google.common.hash.HashCode: HashCode fromLong(long)>

    private static final long serialVersionUID = 0;
  }

  /**
   * Creates a 64-bit {@code HashCode} representation of the given long value. The underlying bytes
   * are interpreted in little endian order.
   *
   * @since 15.0 (since 12.0 in HashCodes)
   */
  public static HashCode fromLong(long hash) {
    return new LongHashCode(hash);
  }

  private static final class LongHashCode extends HashCode implements Serializable {
    final long hash;

    LongHashCode(long hash) {
      this.hash = hash;
    }

