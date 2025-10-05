// Source-based slice around line 45
// Method: <com.google.common.collect.CollectPreconditions: long checkNonnegative(long,String)>

  @CanIgnoreReturnValue
  static int checkNonnegative(int value, String name) {
    if (value < 0) {
      throw new IllegalArgumentException(name + " cannot be negative but was: " + value);
    }
    return value;
  }

  @CanIgnoreReturnValue
  static long checkNonnegative(long value, String name) {
    if (value < 0) {
      throw new IllegalArgumentException(name + " cannot be negative but was: " + value);
    }
    return value;
  }

  static void checkPositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be positive but was: " + value);
    }
