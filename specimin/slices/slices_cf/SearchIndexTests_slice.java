    @SearchIndexFor("a") int y = 7;
    @LTLengthOf("a") int x = i;
  }

  void test2(int[] a, @SearchIndexFor("#1") int xyz) {
    if (0 > xyz) {
      @NegativeIndexFor("a") int w = xyz;
      @NonNegative int y = ~xyz;
      @LTEqLengthOf("a") int z = ~xyz;
    }
  }

  void test3(int[] a, @SearchIndexFor("#1") int xyz) {
    if (-1 >= xyz) {
      @NegativeIndexFor("a") int w = xyz;
      @NonNegative int y = ~xyz;
      @LTEqLengthOf("a") int z = ~xyz;
    }
  }

  void test4(int[] a, @SearchIndexFor("#1") int xyz) {
