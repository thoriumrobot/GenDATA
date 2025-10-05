// Source-based slice around line 129
// Method: <com.google.common.hash.Hashing: HashFunction murmur3_32()>

   * <p>The C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A), which however does not
   * have the bug.
   *
   * @deprecated This implementation produces incorrect hash values from the {@link
   *     HashFunction#hashString} method if the string contains non-BMP characters. Use {@link
   *     #murmur3_32_fixed()} instead.
   */
  @Deprecated
  @SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
  public static HashFunction murmur3_32() {
    return Murmur3_32HashFunction.MURMUR3_32;
  }

  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">32-bit murmur3
   * algorithm, x86 variant</a> (little-endian variant), using the given seed value.
   *
   * <p>The exact C++ equivalent is the MurmurHash3_x86_32 function (Murmur3A).
   *
