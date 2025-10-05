// Source-based slice around line 30
// Method: <com.google.common.math.MathPreconditions: int checkPositive(String,int)>


/**
 * A collection of preconditions for math functions.
 *
 * @author Louis Wasserman
 */
@GwtCompatible
final class MathPreconditions {
  @CanIgnoreReturnValue
  static int checkPositive(String role, int x) {
    if (x <= 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be > 0");
    }
    return x;
  }

  @CanIgnoreReturnValue
  static long checkPositive(String role, long x) {
    if (x <= 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be > 0");
