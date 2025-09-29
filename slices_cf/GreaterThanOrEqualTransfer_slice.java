  void gte_bad_check(int[] a) {
    if (a.length >= 1) {
      // :: error: (assignment)
      int @MinLen(2) [] b = a;
    }
  }
