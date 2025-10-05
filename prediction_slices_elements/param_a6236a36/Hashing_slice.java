// Source-based slice around line 627
// Method: <com.google.common.hash.Hashing: int consistentHash(long,int)>

   *       no way for you to specify which of the three buckets is disappearing. Thus, if your
   *       buckets change from {@code [alpha, bravo, charlie]} to {@code [bravo, charlie]}, it will
   *       assign all the old {@code alpha} traffic to {@code bravo} and all the old {@code bravo}
   *       traffic to {@code charlie}, rather than letting {@code bravo} keep its traffic.
   * </ul>
   *
   * <p>See the <a href="http://en.wikipedia.org/wiki/Consistent_hashing">Wikipedia article on
   * consistent hashing</a> for more information.
   */
  public static int consistentHash(long input, int buckets) {
    checkArgument(buckets > 0, "buckets must be positive: %s", buckets);
    LinearCongruentialGenerator generator = new LinearCongruentialGenerator(input);
    int candidate = 0;
    int next;

    // Jump from bucket to bucket until we go out of range
    while (true) {
      next = (int) ((candidate + 1) / generator.nextDouble());
      if (next >= 0 && next < buckets) {
        candidate = next;
