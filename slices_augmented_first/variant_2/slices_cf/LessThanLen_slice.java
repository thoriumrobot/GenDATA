    @Positive
  public static void m2(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[shorter.length * 2];
    @Positive
    int i = 0;
        while (i < shorter.length) {
            @Positive
      longer[i] = shorter[i];
    @Positive
            i++;
        }
    @Positive
  }

    @Positive
  public static void m3(int[] shorter) {
    @Positive
    int[] longer = new int[shorter.length + 1];
    @Positive
    int i = 0;
        while (i < shorter.length) {
            @Positive
      longer[i] = shorter[i];
    @Positive
            i++;
        }
    @Positive
  }
