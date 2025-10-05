// Source-based slice around line 47
// Method: <com.google.common.hash.HashCode: int asInt()>

  /** Returns the number of bits in this hash code; a positive multiple of 8. */
  public abstract int bits();

  /**
   * Returns the first four bytes of {@linkplain #asBytes() this hashcode's bytes}, converted to an
   * {@code int} value in little-endian order.
   *
   * @throws IllegalStateException if {@code bits() < 32}
   */
  public abstract int asInt();

  /**
   * Returns the first eight bytes of {@linkplain #asBytes() this hashcode's bytes}, converted to a
   * {@code long} value in little-endian order.
   *
   * @throws IllegalStateException if {@code bits() < 64}
   */
  public abstract long asLong();

  /**
