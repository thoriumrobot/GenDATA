    @Positive
  public static void m4(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    int k = i;
    @Positive
    if (k >= j) {
    @Positive
      @IndexFor("a") int y = k;
    @Positive
    }
    @Positive
    k = i;
        while (k >= j) {
            @Positive
      @IndexFor("a") int x = k;
    @Positive
            k -= j;
        }
    @Positive
  }
