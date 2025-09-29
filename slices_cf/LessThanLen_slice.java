  public static void m3(int[] shorter) {
    @Positive
    int[] longer = new int[shorter.length + 1];
    @Positive
    for (int i = 0; i < shorter.length; i++) {
      longer[i] = shorter[i];
    }
  }

  public static void m4(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[shorter.length * 1];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    @LTEqLengthOf("longer") int y = shorter.length;
  }

  public static void m5(int[] shorter) {
    // :: error: (array.length.negative)
    @Positive
    int[] longer = new int[shorter.length * -1];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("longer") int y = shorter.length;
  }
