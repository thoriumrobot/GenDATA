    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (unary.increment)
      offset++;
    }
  }

    @Positive
  public void test1b(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (compound.assignment)
      offset += 1;
    }
  }

    @Positive
  public void test1c(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (compound.assignment)
      offset2 += offset;
    }
  }

  public void test2(int[] a, int[] array) {
    @Positive
    int offset = array.length - 1;
    @Positive
    int offset2 = array.length - 1;

    while (flag) {
      offset++;
      offset2 += offset;
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = offset;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int y = offset2;
  }

    @Positive
  public void test3(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      offset--;
      // :: error: (compound.assignment)
      offset2 -= offset;
    }
  }

    @Positive
  public void test4(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (unary.increment)
      offset++;
      // :: error: (compound.assignment)
      offset += 1;
      // :: error: (compound.assignment)
      offset2 += offset;
    }
  }

  public void test4(int[] src) {
    @Positive
    int patternLength = src.length;
    @Positive
    int[] optoSft = new int[patternLength];
    @Positive
    for (int i = patternLength; i > 0; i--) {}
  }

  public void test5(
      int[] a,
    @Positive
      @LTLengthOf(value = "#1", offset = "-1000") int offset,
    @Positive
      @LTLengthOf("#1") int offset2) {
    @NonNegative
    int otherOffset = offset;
    while (flag) {
      otherOffset += 1;
      // :: error: (unary.increment)
      offset++;
      // :: error: (compound.assignment)
      offset += 1;
      // :: error: (compound.assignment)
      offset2 += offset;
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "#1", offset = "-1000") int x = otherOffset;
  }
