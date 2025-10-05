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

    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    int i = end - 1;
        while (i >= start) {
            @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
            i--;
        }
    @Positive
    return -1;
    @Positive
  }

    @Positive
  GuavaPrimitives(short @MinLen(1) [] array) {
    @Positive
    this(array, 0, array.length);
    @Positive
  }

    @Positive
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
    @Positive
  )
    @Positive
  GuavaPrimitives(
    @Positive
      short @MinLen(1) [] array,
    @Positive
      @IndexFor("#1") @LessThan("#3") int start,
    @Positive
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    @Positive
    this.array = array;
    @Positive
    this.start = start;
    @Positive
    this.end = end;
    @Positive
  }

    @Positive
  public @Positive @LTLengthOf(
    @Positive
      value = {"this", "array"},
    @Positive
      offset = {"-1", "start - 1"}) int
    @Positive
      size() { // INDEX: Annotation on a public method refers to private member.
    @Positive
    return end - start;
    @Positive
  }
