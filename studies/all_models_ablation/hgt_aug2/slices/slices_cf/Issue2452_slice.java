    @Positive
  void m2() {
    @Positive
    int[] arr = {1, 2, 3};
    @Positive
    @LTEqLengthOf({"arr"}) int a = Array.getLength(arr);
    @Positive
  }
