// Source-based slice around line 559
// Method: <com.google.common.hash.Hashing: HashFunction fingerprint2011()>

   * combination is that CityHash has a bunch of special cases for short strings that don't need to
   * be replicated here. The result will never be 0 or 1.
   *
   * <p>This function is best understood as a <a
   * href="https://en.wikipedia.org/wiki/Fingerprint_(computing)">fingerprint</a> rather than a true
   * <a href="https://en.wikipedia.org/wiki/Hash_function">hash function</a>.
   *
   * @since 31.1
   */
  public static HashFunction fingerprint2011() {
    return Fingerprint2011.FINGERPRINT_2011;
  }

  /**
   * Assigns to {@code hashCode} a "bucket" in the range {@code [0, buckets)}, in a uniform manner
   * that minimizes the need for remapping as {@code buckets} grows. That is, {@code
   * consistentHash(h, n)} equals:
   *
   * <ul>
   *   <li>{@code n - 1}, with approximate probability {@code 1/n}
