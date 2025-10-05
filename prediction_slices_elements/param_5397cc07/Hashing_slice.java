// Source-based slice around line 146
// Method: <com.google.common.hash.Hashing: HashFunction murmur3_32_fixed(int)>

   *
   * <p>The exact C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A).
   *
   * <p>This method is called {@code murmur3_32_fixed} because it fixes a bug in the {@code
   * HashFunction} returned by the original {@code murmur3_32} method.
   *
   * @since 31.0
   */
  @SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
  public static HashFunction murmur3_32_fixed(int seed) {
    return new Murmur3_32HashFunction(seed, /* supplementaryPlaneFix= */ true);
  }

  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">32-bit murmur3
   * algorithm, x86 variant</a> (little-endian variant), using a seed value of zero.
   *
   * <p>The exact C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A).
   *
