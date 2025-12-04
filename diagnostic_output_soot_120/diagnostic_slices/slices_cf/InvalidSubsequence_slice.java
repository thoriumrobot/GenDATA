    @Positive
  void assignA(int[] d) {
    // :: error: (to.not.ltel)
    @Positive
    a = d;
    @Positive
  }

    @Positive
  void assignB(int[] d) {
    // :: error: (from.gt.to) :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    @Positive
    b = d;
    @Positive
  }
