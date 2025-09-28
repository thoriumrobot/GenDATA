  void bug(@IntRange(to = 42) int a, @IntVal(1) int c) {
    // :: error: (assignment)
    @LessThan("c") int x = a;
  }
