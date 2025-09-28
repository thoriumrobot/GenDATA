  void eq_bad_check(int[] a) {
    if (1 == a.length) {
      // :: error: (assignment)
      int @MinLen(2) [] b = a;
    }
  }
