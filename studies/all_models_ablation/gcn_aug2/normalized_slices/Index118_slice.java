    @Positive
  public static void foo(String @ArrayLen(4) [] args) {
    @Positive
    for (int i = 1; i <= 3; i++) {
    @Positive
      @IntRange(from = 1, to = 3) int x = i;
    @Positive
      System.out.println(args[i]);
    @Positive
    }
    @Positive
  }
