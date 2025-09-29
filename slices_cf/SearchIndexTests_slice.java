  public void test(short[] a, short instant) {
    int i = Arrays.binarySearch(a, instant);
    @NonNegative
    @SearchIndexFor("a") int z = i;
    // :: error: (assignment)
    @NonNegative
    @SearchIndexFor("a") int y = 7;
    @Positive
    @LTLengthOf("a") int x = i;
  }

    @NonNegative
  void test2(int[] a, @SearchIndexFor("#1") int xyz) {
    if (0 > xyz) {
    @NonNegative
      @NegativeIndexFor("a") int w = xyz;
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    }
  }

    @NonNegative
  void test3(int[] a, @SearchIndexFor("#1") int xyz) {
    if (-1 >= xyz) {
    @NonNegative
      @NegativeIndexFor("a") int w = xyz;
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    }
  }

    @NonNegative
  void test4(int[] a, @SearchIndexFor("#1") int xyz) {
    if (xyz < 0) {
    @NonNegative
      @NegativeIndexFor("a") int w = xyz;
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    }
  }

    @NonNegative
  void test5(int[] a, @SearchIndexFor("#1") int xyz) {
    if (xyz <= -1) {
    @NonNegative
      @NegativeIndexFor("a") int w = xyz;
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    }
  }

  void subtyping1(
    @NonNegative
      @SearchIndexFor({"#3", "#4"}) int x, @NegativeIndexFor("#3") int y, int[] a, int[] b) {
    // :: error: (assignment)
    @NonNegative
    @SearchIndexFor({"a", "b"}) int z = y;
    @NonNegative
    @SearchIndexFor("a") int w = y;
    @NonNegative
    @SearchIndexFor("b") int p = x;
    // :: error: (assignment)
    @NonNegative
    @NegativeIndexFor({"a", "b"}) int q = x;
  }
