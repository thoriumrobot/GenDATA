// Source-based slice around line 1365
// Method: <com.google.common.math.LongMath: long saturatedAbs(long)>

   *       when passed {@code Long.MIN_VALUE}
   * </ul>
   *
   * <p>Note that if your only goal is to turn a well-distributed `long` (such as a random number)
   * into a well-distributed nonnegative number, the most even distribution is achieved not by this
   * method or other absolute value methods, but by {@code x & Long.MAX_VALUE}.
   *
   * @since 33.5.0
   */
  public static long saturatedAbs(long x) {
    return (x == Long.MIN_VALUE) ? Long.MAX_VALUE : Math.abs(x);
  }

  private LongMath() {}
}
