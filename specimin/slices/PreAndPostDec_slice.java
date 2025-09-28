  void pre1(int[] args) {
    int ii = 0;
    while ((ii < args.length)) {
      // :: error: (array.access.unsafe.high)
      int m = args[++ii];
    }
  }

  void pre2(int[] args) {
    int ii = 0;
    while ((ii < args.length)) {
      ii++;
      // :: error: (array.access.unsafe.high)
      int m = args[ii];
    }
  }
