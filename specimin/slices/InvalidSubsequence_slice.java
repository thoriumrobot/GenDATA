  void assignB(int[] d) {
    // :: error: (from.gt.to) :: error: (from.not.nonnegative) :: error: (to.not.ltel)
    b = d;
  }

  void assignC(int[] d) {
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    c = d;
  }

  void assignE(int[] d) {
    // :: error: (from.gt.to) :: error: (to.not.ltel)
    e = d;
  }
