    @Positive
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
    @Positive
    return indexOf(array, target, 0, array.length);
    @Positive
  }

    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    int i = start;
        while (i < end) {
            @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
            i++;
        }
    @Positive
    return -1;
    @Positive
  }
