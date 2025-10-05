
    @Positive
  int[] a;
    @Positive
  int[] b;
    @Positive
  int c, d;

    @Positive
  void badParam(
      // :: error: (different.length.sequences.offsets)
    @Positive
              value = {"a", "b"},
    @Positive
              offset = {"c"})
    @Positive
          int x) {}

    @Positive
  void badParam2(
      // :: error: (different.length.sequences.offsets)
