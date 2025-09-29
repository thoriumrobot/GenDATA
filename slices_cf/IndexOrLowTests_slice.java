  void test() {

    if (index != -1) {
      array[index] = 1;
    }

    @NonNegative
    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    array[y] = 1;
    if (y < array.length) {
      array[y] = 1;
    }
    // :: error: (assignment)
    index = array.length;
  }

    @Positive
  void test2(@LTLengthOf("array") @GTENegativeOne int param) {
    index = array.length - 1;
    @Positive
    @LTLengthOf("array") @GTENegativeOne int x = index;
    index = param;
  }
