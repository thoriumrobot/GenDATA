// Source-based slice around line 686
// Method: <com.google.common.collect.Range: String toString(Cut,Cut)>

  /**
   * Returns a string representation of this range, such as {@code "[3..5)"} (other examples are
   * listed in the class documentation).
   */
  @Override
  public String toString() {
    return toString(lowerBound, upperBound);
  }

  private static String toString(Cut<?> lowerBound, Cut<?> upperBound) {
    StringBuilder sb = new StringBuilder(16);
    lowerBound.describeAsLowerBound(sb);
    sb.append("..");
    upperBound.describeAsUpperBound(sb);
    return sb.toString();
  }

  // We declare accessors so that we can use method references like `Range::lowerBound`.

  Cut<C> lowerBound() {
