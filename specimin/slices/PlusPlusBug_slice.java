  void test(@LTLengthOf("array") int x) {
    // :: error: (unary.increment)
    x++;
    // :: error: (unary.increment)
    ++x;
    // :: error: (assignment)
    x = x + 1;
  }
