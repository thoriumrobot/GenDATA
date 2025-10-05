// Source-based slice around line 481
// Method: <com.google.common.hash.Hashing: HashFunction adler32()>

   * <p>To get the {@code long} value equivalent to {@link Checksum#getValue()} for a {@code
   * HashCode} produced by this function, use {@link HashCode#padToLong()}.
   *
   * <p>This function is best understood as a <a
   * href="https://en.wikipedia.org/wiki/Checksum">checksum</a> rather than a true <a
   * href="https://en.wikipedia.org/wiki/Hash_function">hash function</a>.
   *
   * @since 14.0
   */
  public static HashFunction adler32() {
    return ChecksumType.ADLER_32.hashFunction;
  }

  @Immutable
  enum ChecksumType implements ImmutableSupplier<Checksum> {
    CRC_32("Hashing.crc32()") {
      @Override
      public Checksum get() {
        return new CRC32();
      }
