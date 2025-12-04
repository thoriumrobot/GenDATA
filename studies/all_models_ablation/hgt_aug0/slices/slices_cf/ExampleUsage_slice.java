    @Positive
  void safe_loop_spooky() {
    @Positive
    int[] arr = new int[5];
    @Positive
    int k;
    @Positive
    for (int i = -1; i < 4; ) {
    @Positive
      i++;
    @Positive
      k = arr[i];
    @Positive
    }
    @Positive
  }
