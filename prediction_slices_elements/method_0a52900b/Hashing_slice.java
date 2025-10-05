// Source-based slice around line 175
// Method: <com.google.common.hash.Hashing: HashFunction murmur3_128(int)>


  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">128-bit murmur3
   * algorithm, x64 variant</a> (little-endian variant), using the given seed value.
   *
   * <p>The exact C++ equivalent is the MurmurHash3_x64_128 function (Murmur3F).
   */
  @SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
  public static HashFunction murmur3_128(int seed) {
    return new Murmur3_128HashFunction(seed);
  }

  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">128-bit murmur3
   * algorithm, x64 variant</a> (little-endian variant), using a seed value of zero.
   *
   * <p>The exact C++ equivalent is the MurmurHash3_x64_128 function (Murmur3F).
   */
