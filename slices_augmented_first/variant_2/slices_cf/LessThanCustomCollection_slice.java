    @Positive
  private LessThanCustomCollection(
    @Positive
      int[] array, @IndexOrHigh("#1") @LessThan("#1 + 3") int start, @IndexOrHigh("#1") int end) {
    @Positive
    this.array = array;
    // can't est. that end - start is the length of this.
    // :: error: (assignment)
    @Positive
    this.end = end;
    // start is @LessThan(1 + end) but should be @LessThan(this.1 + end)
    // :: error: (assignment)
    @Positive
    this.start = start;
    @Positive
  }

    @Positive
  public @LengthOf("this") int length() {
    @Positive
    return end - start;
    @Positive
  }
