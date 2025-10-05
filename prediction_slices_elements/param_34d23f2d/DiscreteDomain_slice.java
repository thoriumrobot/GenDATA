// Source-based slice around line 263
// Method: <com.google.common.collect.DiscreteDomain: C offset(C,long)>

  /** Private constructor for built-in DiscreteDomains supporting fast offset. */
  private DiscreteDomain(boolean supportsFastOffset) {
    this.supportsFastOffset = supportsFastOffset;
  }

  /**
   * Returns, conceptually, "origin + distance", or equivalently, the result of calling {@link
   * #next} on {@code origin} {@code distance} times.
   */
  C offset(C origin, long distance) {
    C current = origin;
    checkNonnegative(distance, "distance");
    for (long i = 0; i < distance; i++) {
      current = next(current);
      if (current == null) {
        throw new IllegalArgumentException(
            "overflowed computing offset(" + origin + ", " + distance + ")");
      }
    }
    return current;
