// Source-based slice around line 197
// Method: <com.google.common.hash.Hashing: HashFunction sipHash24()>

    return Murmur3_128HashFunction.MURMUR3_128;
  }

  /**
   * Returns a hash function implementing the <a href="https://131002.net/siphash/">64-bit
   * SipHash-2-4 algorithm</a> using a seed value of {@code k = 00 01 02 ...}.
   *
   * @since 15.0
   */
  public static HashFunction sipHash24() {
    return SipHashFunction.SIP_HASH_24;
  }

  /**
   * Returns a hash function implementing the <a href="https://131002.net/siphash/">64-bit
   * SipHash-2-4 algorithm</a> using the given seed.
   *
   * @since 15.0
   */
  public static HashFunction sipHash24(long k0, long k1) {
