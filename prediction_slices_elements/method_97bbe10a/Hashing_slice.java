// Source-based slice around line 207
// Method: <com.google.common.hash.Hashing: HashFunction sipHash24(long,long)>

    return SipHashFunction.SIP_HASH_24;
  }

  /**
   * Returns a hash function implementing the <a href="https://131002.net/siphash/">64-bit
   * SipHash-2-4 algorithm</a> using the given seed.
   *
   * @since 15.0
   */
  public static HashFunction sipHash24(long k0, long k1) {
    return new SipHashFunction(2, 4, k0, k1);
  }

  /**
   * Returns a hash function implementing the MD5 hash algorithm (128 hash bits).
   *
   * @deprecated If you must interoperate with a system that requires MD5, then use this method,
   *     despite its deprecation. But if you can choose your hash function, avoid MD5, which is
   *     neither fast nor secure. As of January 2017, we suggest:
   *     <ul>
