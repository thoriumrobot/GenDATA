// Source-based slice around line 661
// Method: <com.google.common.graph.Graphs: int checkPositive(int)>

  }

  @CanIgnoreReturnValue
  static long checkNonNegative(long value) {
    checkArgument(value >= 0, "Not true that %s is non-negative.", value);
    return value;
  }

  @CanIgnoreReturnValue
  static int checkPositive(int value) {
    checkArgument(value > 0, "Not true that %s is positive.", value);
    return value;
  }

  @CanIgnoreReturnValue
  static long checkPositive(long value) {
    checkArgument(value > 0, "Not true that %s is positive.", value);
    return value;
  }

