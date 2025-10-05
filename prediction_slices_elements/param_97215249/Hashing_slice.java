// Source-based slice around line 593
// Method: <com.google.common.hash.Hashing: int consistentHash(HashCode,int)>

   *       no way for you to specify which of the three buckets is disappearing. Thus, if your
   *       buckets change from {@code [alpha, bravo, charlie]} to {@code [bravo, charlie]}, it will
   *       assign all the old {@code alpha} traffic to {@code bravo} and all the old {@code bravo}
   *       traffic to {@code charlie}, rather than letting {@code bravo} keep its traffic.
   * </ul>
   *
   * <p>See the <a href="http://en.wikipedia.org/wiki/Consistent_hashing">Wikipedia article on
   * consistent hashing</a> for more information.
   */
  public static int consistentHash(HashCode hashCode, int buckets) {
    return consistentHash(hashCode.padToLong(), buckets);
  }

  /**
   * Assigns to {@code input} a "bucket" in the range {@code [0, buckets)}, in a uniform manner that
   * minimizes the need for remapping as {@code buckets} grows. That is, {@code consistentHash(h,
   * n)} equals:
   *
   * <ul>
   *   <li>{@code n - 1}, with approximate probability {@code 1/n}
