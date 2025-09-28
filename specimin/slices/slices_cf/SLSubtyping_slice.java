  void subtype(int @SameLen("#2") [] a, int[] b) {
    int @SameLen({"a", "b"}) [] c = a;

    // :: error: (assignment)
    int @SameLen("c") [] q = {1, 2};
    int @SameLen("c") [] d = q;

    // :: error: (assignment)
    int @SameLen("f") [] e = a;
  }
