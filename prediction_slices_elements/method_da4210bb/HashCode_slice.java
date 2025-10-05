// Source-based slice around line 55
// Method: <com.google.common.hash.HashCode: long asLong()>

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
   * If this hashcode has enough bits, returns {@code asLong()}, otherwise returns a {@code long}
   * value with {@code asBytes()} as the least-significant bytes and {@code 0x00} as the remaining
   * most-significant bytes.
   *
   * @since 14.0 (since 11.0 as {@code Hashing.padToLong(HashCode)})
   */
  public abstract long padToLong();

