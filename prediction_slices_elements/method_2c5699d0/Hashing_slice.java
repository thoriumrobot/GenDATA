// Source-based slice around line 535
// Method: <com.google.common.hash.Hashing: HashFunction farmHashFingerprint64()>

   * farmhash::Fingerprint64() would for the same input (when compared using {@link
   * com.google.common.primitives.UnsignedLongs}'s encoding of 64-bit unsigned numbers).
   *
   * <p>This function is best understood as a <a
   * href="https://en.wikipedia.org/wiki/Fingerprint_(computing)">fingerprint</a> rather than a true
   * <a href="https://en.wikipedia.org/wiki/Hash_function">hash function</a>.
   *
   * @since 20.0
   */
  public static HashFunction farmHashFingerprint64() {
    return FarmHashFingerprint64.FARMHASH_FINGERPRINT_64;
  }

  /**
   * Returns a hash function implementing the Fingerprint2011 hashing function (64 hash bits).
   *
   * <p>This is designed for generating persistent fingerprints of strings. It isn't
   * cryptographically secure, but it produces a high-quality hash with few collisions. Fingerprints
   * generated using this are byte-wise identical to those created using the C++ version, but note
   * that this uses unsigned integers (see {@link com.google.common.primitives.UnsignedInts}).
