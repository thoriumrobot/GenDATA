// Source-based slice around line 47
// Method: com.google.common.hash.SipHashFunction.k1

  static final HashFunction SIP_HASH_24 =
      new SipHashFunction(2, 4, 0x0706050403020100L, 0x0f0e0d0c0b0a0908L);

  // The number of compression rounds.
  private final int c;
  // The number of finalization rounds.
  private final int d;
  // Two 64-bit keys (represent a single 128-bit key).
  private final long k0;
  private final long k1;

  /**
   * @param c the number of compression rounds (must be positive)
   * @param d the number of finalization rounds (must be positive)
   * @param k0 the first half of the key
   * @param k1 the second half of the key
   */
  SipHashFunction(int c, int d, long k0, long k1) {
    checkArgument(
        c > 0, "The number of SipRound iterations (c=%s) during Compression must be positive.", c);
