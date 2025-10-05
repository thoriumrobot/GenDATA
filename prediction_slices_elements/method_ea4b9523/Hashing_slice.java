// Source-based slice around line 187
// Method: <com.google.common.hash.Hashing: HashFunction murmur3_128()>


  /**
   * Returns a hash function implementing the <a
   * href="https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp">128-bit murmur3
   * algorithm, x64 variant</a> (little-endian variant), using a seed value of zero.
   *
   * <p>The exact C++ equivalent is the MurmurHash3_x64_128 function (Murmur3F).
   */
  @SuppressWarnings("IdentifierName") // the best we could do for adjacent digit blocks
  public static HashFunction murmur3_128() {
    return Murmur3_128HashFunction.MURMUR3_128;
  }

  /**
   * Returns a hash function implementing the <a href="https://131002.net/siphash/">64-bit
   * SipHash-2-4 algorithm</a> using a seed value of {@code k = 00 01 02 ...}.
   *
   * @since 15.0
   */
  public static HashFunction sipHash24() {
