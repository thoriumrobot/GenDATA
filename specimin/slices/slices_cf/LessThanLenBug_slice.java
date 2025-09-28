  public static void m1(int[] shorter) {
    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @LTLengthOf("longer") int x = shorter.length;
    int i = longer[x];
  }
