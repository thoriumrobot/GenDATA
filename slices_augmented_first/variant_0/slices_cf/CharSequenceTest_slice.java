    @Positive
  void minLenCharSequence() {
    // :: error: (assignment)
    @Positive
    @MinLen(10) CharSequence sb = new StringBuilder("0123456789");
    @Positive
  }

  // Tests the subSequence method
    @Positive
  void testSubSequence() {
    // Local variable used because of https://github.com/kelloggm/checker-framework/issues/165
    @Positive
    String str = "0123456789";
    @Positive
    str.subSequence(5, 8);
    // :: error: (argument)
    @Positive
    str.subSequence(5, 13);
    @Positive
  }

  // Dummy method that takes a CharSequence and its index
    @Positive
  void sink(CharSequence cs, @IndexOrHigh("#1") int i) {}

  // Tests passing sequences as CharSequence
    @Positive
  void argumentPassing() {
    @Positive
    String s = "0123456789";
    @Positive
    sink(s, 8);
    @Positive
    StringBuilder sb = new StringBuilder("0123456789");
    // :: error: (argument)
    @Positive
    sink(sb, 8);
    @Positive
  }

  // Tests forwardning sequences as CharSequence
    @Positive
  void agumentForwarding(String s, @IndexOrHigh("#1") int i) {
    @Positive
    sink(s, i);
    @Positive
  }

  // Tests concatenation of CharSequence and String
    @Positive
  void concat() {
    @Positive
    CharSequence a = "a";
    @Positive
    @StringVal({"nullb", "ab"}) CharSequence ab = a + "b";
    @Positive
    sink(ab, 2);
    @Positive
  }
