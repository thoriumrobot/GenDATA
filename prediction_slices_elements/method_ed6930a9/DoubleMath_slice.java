// Source-based slice around line 384
// Method: <com.google.common.math.DoubleMath: int fuzzyCompare(double,double,double)>

   * b)}. In particular, like {@link Double#compare(double, double)}, it treats all NaN values as
   * equal and greater than all other values (including {@link Double#POSITIVE_INFINITY}).
   *
   * <p>This is <em>not</em> a total ordering and is <em>not</em> suitable for use in {@link
   * Comparable#compareTo} implementations. In particular, it is not transitive.
   *
   * @throws IllegalArgumentException if {@code tolerance} is {@code < 0} or NaN
   * @since 13.0
   */
  public static int fuzzyCompare(double a, double b, double tolerance) {
    if (fuzzyEquals(a, b, tolerance)) {
      return 0;
    } else if (a < b) {
      return -1;
    } else if (a > b) {
      return 1;
    } else {
      return Boolean.compare(Double.isNaN(a), Double.isNaN(b));
    }
  }
