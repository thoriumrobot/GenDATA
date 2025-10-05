// Source-based slice around line 64
// Method: <com.google.common.hash.HashCode: long padToLong()>

  public abstract long asLong();

  /**
   * If this hashcode has enough bits, returns {@code asLong()}, otherwise returns a {@code long}
   * value with {@code asBytes()} as the least-significant bytes and {@code 0x00} as the remaining
   * most-significant bytes.
   *
   * @since 14.0 (since 11.0 as {@code Hashing.padToLong(HashCode)})
   */
  public abstract long padToLong();

  /**
   * Returns the value of this hash code as a byte array. The caller may modify the byte array;
   * changes to it will <i>not</i> be reflected in this {@code HashCode} object or any other arrays
   * returned by this method.
   */
  // TODO(user): consider ByteString here, when that is available
  public abstract byte[] asBytes();

  /**
