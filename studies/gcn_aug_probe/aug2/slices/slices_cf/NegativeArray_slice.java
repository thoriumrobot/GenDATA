    @Positive
  public static void negativeArray(@GTENegativeOne int len) {
    // :: error: (array.length.negative)
    @Positive
    int[] arr = new int[len];
    @Positive
  }
