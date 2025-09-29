  void lt_bad_check(int[] a) {
    if (0 < a.length) {
      // :: error: (assignment)
      int @MinLen(2) [] b = a;
    }
  }
