  void callTest1(int x) {
    test1(0);
    test1(1);
    test1(2);
    test1(array.length);
    // :: error: (argument)
    test1(array.length - 1);
    if (array.length > x) {
      // :: error: (argument)
      test1(x);
    }

    if (array.length == x) {
      test1(x);
    }
  }
