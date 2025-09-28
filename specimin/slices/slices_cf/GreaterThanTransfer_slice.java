  void gt_bad_check(int[] a) {
    if (a.length > 0) {
      // :: error: (assignment)
      int @MinLen(2) [] b = a;
    }
  }
