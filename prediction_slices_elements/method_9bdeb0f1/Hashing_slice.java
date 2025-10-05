// Source-based slice around line 163
// Method: <com.google.common.hash.Hashing: HashFunction murmur3_32_fixed()>

   *
   * <p>The exact C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A).
   *
   * <p>This method is called {@code murmur3_32_fixed} because it fixes a bug in the {@code
   * HashFunction} returned by the original {@code murmur3_32} method.
   *
   * @since 31.0
   */
  @SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
  public static HashFunction murmur3_32_fixed() {
    return Murmur3_32HashFunction.MURMUR3_32_FIXED;
  }

  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">128-bit murmur3
   * algorithm, x64 variant</a> (little-endian variant), using the given seed value.
   *
   * <p>The exact C++ equivalent is the MurmurHash3_x64_128 function (Murmur3F).
   */
