    @Positive
  public void test(short[] a, short instant) {
    @Positive
    int i = Arrays.binarySearch(a, instant);
    @Positive
    @SearchIndexFor("a") int z = i;
    // :: error: (assignment)
    @Positive
    @SearchIndexFor("a") int y = 7;
    @Positive
    @LTLengthOf("a") int x = i;
    @Positive
  }

    @Positive
  void test2(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (0 > xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test3(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (-1 >= xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test4(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz < 0) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }
