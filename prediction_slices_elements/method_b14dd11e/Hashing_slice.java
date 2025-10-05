// Source-based slice around line 110
// Method: <com.google.common.hash.Hashing: HashFunction murmur3_32(int)>

   * <p>The C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A), which however does not
   * have the bug.
   *
   * @deprecated This implementation produces incorrect hash values from the {@link
   *     HashFunction#hashString} method if the string contains non-BMP characters. Use {@link
   *     #murmur3_32_fixed(int)} instead.
   */
  @Deprecated
  @SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
  public static HashFunction murmur3_32(int seed) {
    return new Murmur3_32HashFunction(seed, /* supplementaryPlaneFix= */ false);
  }

  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">32-bit murmur3
   * algorithm, x86 variant</a> (little-endian variant), using the given seed value, <b>with a known
   * bug</b> as described in the deprecation text.
   *
   * <p>The C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A), which however does not
