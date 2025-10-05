// Source-based slice around line 411
// Method: <com.google.common.hash.Hashing: HashFunction crc32c()>

   * Returns a hash function implementing the CRC32C checksum algorithm (32 hash bits) as described
   * by RFC 3720, Section 12.1.
   *
   * <p>This function is best understood as a <a
   * href="https://en.wikipedia.org/wiki/Checksum">checksum</a> rather than a true <a
   * href="https://en.wikipedia.org/wiki/Hash_function">hash function</a>.
   *
   * @since 18.0
   */
  public static HashFunction crc32c() {
    return Crc32CSupplier.HASH_FUNCTION;
  }

  @Immutable
  private enum Crc32CSupplier implements ImmutableSupplier<HashFunction> {
    @J2ObjCIncompatible
    JAVA_UTIL_ZIP {
      @Override
      public HashFunction get() {
        return ChecksumType.CRC_32C.hashFunction;
