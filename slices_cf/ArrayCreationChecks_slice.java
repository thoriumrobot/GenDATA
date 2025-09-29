  void test1(@Positive int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @NonNegative
    @IndexFor("newArray") int i = x;
    @NonNegative
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @NonNegative
    @IndexFor("newArray") int i = x;
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
    int[] newArray = new int[x + y];
    @NonNegative
    @IndexOrHigh("newArray") int i = x;
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }

  void test4(@GTENegativeOne int x, @NonNegative int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    @Positive
    @LTEqLengthOf("newArray") int i = x;
    // :: error: (assignment)
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }

  void test5(@GTENegativeOne int x, @GTENegativeOne int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @NonNegative
    @IndexOrHigh("newArray") int i = x;
    // :: error: (assignment)
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }

  void test6(int x, int y) {
    // :: error: (array.length.negative)
    int[] newArray = new int[x + y];
    // :: error: (assignment)
    @NonNegative
    @IndexFor("newArray") int i = x;
    // :: error: (assignment)
    @NonNegative
    @IndexOrHigh("newArray") int j = y;
  }
