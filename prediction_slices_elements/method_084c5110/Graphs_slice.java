// Source-based slice around line 655
// Method: <com.google.common.graph.Graphs: long checkNonNegative(long)>

  }

  @CanIgnoreReturnValue
  static int checkNonNegative(int value) {
    checkArgument(value >= 0, "Not true that %s is non-negative.", value);
    return value;
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

