  public static void m5(String @MinLen(1) [] args, String[] otherArray) {
    // :: error: (assignment)
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, otherArray.length);
  }

  public static void m6(String @MinLen(1) [] args) {
    // :: error: (assignment)
    String @MinLen(1) [] args2 = Arrays.copyOf(args, args.length);
  }
