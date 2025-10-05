// Source-based slice around line 465
// Method: <com.google.common.hash.Hashing: HashFunction crc32()>

   * <p>To get the {@code long} value equivalent to {@link Checksum#getValue()} for a {@code
   * HashCode} produced by this function, use {@link HashCode#padToLong()}.
   *
   * <p>This function is best understood as a <a
   * href="https://en.wikipedia.org/wiki/Checksum">checksum</a> rather than a true <a
   * href="https://en.wikipedia.org/wiki/Hash_function">hash function</a>.
   *
   * @since 14.0
   */
  public static HashFunction crc32() {
    return ChecksumType.CRC_32.hashFunction;
  }

  /**
   * Returns a hash function implementing the Adler-32 checksum algorithm (32 hash bits).
   *
   * <p>To get the {@code long} value equivalent to {@link Checksum#getValue()} for a {@code
   * HashCode} produced by this function, use {@link HashCode#padToLong()}.
   *
   * <p>This function is best understood as a <a
