// Source-based slice around line 66
// Method: <com.google.common.hash.Hashing: HashFunction goodFastHash(int)>

   *
   * <p>Repeated calls to this method on the same loaded {@code Hashing} class, using the same value
   * for {@code minimumBits}, will return identically-behaving {@link HashFunction} instances.
   *
   * @param minimumBits a positive integer. This can be arbitrarily large. The returned {@link
   *     HashFunction} instance may use memory proportional to this integer.
   * @return a hash function, described above, that produces hash codes of length {@code
   *     minimumBits} or greater
   */
  public static HashFunction goodFastHash(int minimumBits) {
    int bits = checkPositiveAndMakeMultipleOf32(minimumBits);

    if (bits == 32) {
      return Murmur3_32HashFunction.GOOD_FAST_HASH_32;
    }
    if (bits <= 128) {
      return Murmur3_128HashFunction.GOOD_FAST_HASH_128;
    }

    // Otherwise, join together some 128-bit murmur3s
