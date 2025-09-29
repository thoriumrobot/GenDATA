  @EnsuresLTLengthOf(value = "end", targetValue = "array", offset = "#1 - 1")
    @NonNegative
  public void shiftIndex(@NonNegative int x) {
    int newEnd = end - x;
    if (newEnd < 0) throw new RuntimeException();
    end = newEnd;
  }

    @NonNegative
  public void useShiftIndex(@NonNegative int x) {
    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfIf(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
    @NonNegative
  public boolean tryShiftIndex(@NonNegative int x) {
    int newEnd = end - x;
    if (newEnd < 0) {
      return false;
    }
    end = newEnd;
    return true;
  }

    @NonNegative
  public void useTryShiftIndex(@NonNegative int x) {
    if (tryShiftIndex(x)) {
      Arrays.fill(array, end, end + x, null);
    }
  }
