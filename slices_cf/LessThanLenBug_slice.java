  public static void m1(int[] shorter) {
    @Positive
    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    int i = longer[x];
  }
