    @Positive
  public static void bug(int[] a, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("a") int k = (i + j) / 2;
    @Positive
  }
