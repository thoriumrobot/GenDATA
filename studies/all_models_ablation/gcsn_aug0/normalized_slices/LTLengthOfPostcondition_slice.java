    @Positive
  public void shiftIndex(@NonNegative int x) {
    @Positive
    int newEnd = end - x;
    @Positive
    if (newEnd < 0) throw new RuntimeException();
    @Positive
    end = newEnd;
    @Positive
  }

    @Positive
  public void useShiftIndex(@NonNegative int x) {
    // :: error: (argument)
    @Positive
    Arrays.fill(array, end, end + x, null);
    @Positive
    shiftIndex(x);
    @Positive
    Arrays.fill(array, end, end + x, null);
    @Positive
  }
