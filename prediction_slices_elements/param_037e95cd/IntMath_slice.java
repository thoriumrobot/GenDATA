// Source-based slice around line 756
// Method: <com.google.common.math.IntMath: int saturatedAbs(int)>

   *       when passed {@code Integer.MIN_VALUE}
   * </ul>
   *
   * <p>Note that if your only goal is to turn a well-distributed `int` (such as a random number or
   * hash code) into a well-distributed nonnegative number, the most even distribution is achieved
   * not by this method or other absolute value methods, but by {@code x & Integer.MAX_VALUE}.
   *
   * @since 33.5.0
   */
  public static int saturatedAbs(int x) {
    return (x == Integer.MIN_VALUE) ? Integer.MAX_VALUE : Math.abs(x);
  }

  private IntMath() {}
}
