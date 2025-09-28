  void gteN1Test(@GTENegativeOne int y) {
    int[] arr = new int[10];
    if (-1 != y) {
      @NonNegative int z = y;
      if (z < 10) {
        int k = arr[z];
      }
    }
  }
