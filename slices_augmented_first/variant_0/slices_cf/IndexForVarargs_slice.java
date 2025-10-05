    @Positive
  String get(@IndexFor("#2") int i, String... varargs) {
    @Positive
    return varargs[i];
    @Positive
  }

    @Positive
  void method(@IndexFor("#2") int i, String[]... varargs) {}

    @Positive
  void m() {
    // :: error: (argument)
    @Positive
    get(1);

    @Positive
    get(1, "a", "b");

    // :: error: (argument)
    @Positive
    get(2, "abc");

    @Positive
    String[] stringArg1 = new String[] {"a", "b"};
    @Positive
    String[] stringArg2 = new String[] {"c", "d", "e"};
    @Positive
    String[] stringArg3 = new String[] {"a", "b", "c"};

    @Positive
    method(1, stringArg1, stringArg2);

    // :: error: (argument)
    @Positive
    method(2, stringArg3);

    @Positive
    get(1, stringArg1);

    // :: error: (argument)
    @Positive
    get(3, stringArg2);
    @Positive
  }
