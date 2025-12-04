    @Positive
  public static void m2(String @MinLen(1) [] args) {
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, args.length);
    @Positive
  }

    @Positive
  public static void m3(String @MinLen(1) ... args) {
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, args.length);
    @Positive
  }
