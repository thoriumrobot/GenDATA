// Source-based slice around line 653
// Method: <com.google.common.hash.Hashing: HashCode combineOrdered(Iterable)>

  /**
   * Returns a hash code, having the same bit length as each of the input hash codes, that combines
   * the information of these hash codes in an ordered fashion. That is, whenever two equal hash
   * codes are produced by two calls to this method, it is <i>as likely as possible</i> that each
   * was computed from the <i>same</i> input hash codes in the <i>same</i> order.
   *
   * @throws IllegalArgumentException if {@code hashCodes} is empty, or the hash codes do not all
   *     have the same bit length
   */
  public static HashCode combineOrdered(Iterable<HashCode> hashCodes) {
    Iterator<HashCode> iterator = hashCodes.iterator();
    checkArgument(iterator.hasNext(), "Must be at least 1 hash code to combine.");
    int bits = iterator.next().bits();
    byte[] resultBytes = new byte[bits / 8];
    for (HashCode hashCode : hashCodes) {
      byte[] nextBytes = hashCode.asBytes();
      checkArgument(
          nextBytes.length == resultBytes.length, "All hashcodes must have the same bit length.");
      for (int i = 0; i < nextBytes.length; i++) {
        resultBytes[i] = (byte) (resultBytes[i] * 37 ^ nextBytes[i]);
