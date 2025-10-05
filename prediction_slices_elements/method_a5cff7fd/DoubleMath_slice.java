// Source-based slice around line 54
// Method: <com.google.common.math.DoubleMath: double roundIntermediate(double,RoundingMode)>

 * @since 11.0
 */
@GwtCompatible
public final class DoubleMath {
  /*
   * This method returns a value y such that rounding y DOWN (towards zero) gives the same result as
   * rounding x according to the specified mode.
   */
  @GwtIncompatible // #isMathematicalInteger, com.google.common.math.DoubleUtils
  static double roundIntermediate(double x, RoundingMode mode) {
    if (!isFinite(x)) {
      throw new ArithmeticException("input is infinite or NaN");
    }
    switch (mode) {
      case UNNECESSARY:
        checkRoundingUnnecessary(isMathematicalInteger(x));
        return x;

      case FLOOR:
        if (x >= 0.0 || isMathematicalInteger(x)) {
