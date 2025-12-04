    @Positive
  public static void m(Object[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("#1") int k = (i + j) >> 1;
    @Positive
  }
