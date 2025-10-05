    @Positive
  void test(@IndexFor("this.arrayLen2") int i) {
    @Positive
    int j = arrayLen2[i];
    @Positive
    int j2 = arrayLen2[1];
    @Positive
  }

    @Positive
  void callTest(int x) {
    @Positive
    test(0);
    @Positive
    test(1);
    // :: error: (argument)
    @Positive
    test(2);
    // :: error: (argument)
    @Positive
    test(3);
    @Positive
    test(arrayLen2.length - 1);
    @Positive
  }
