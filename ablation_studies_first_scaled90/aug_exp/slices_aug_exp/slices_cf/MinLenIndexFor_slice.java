    @Positive
  void test(@IndexFor("this.arrayLen2") int i) {
    @Positive
    int j = arrayLen2[i];
    @Positive
    int j2 = arrayLen2[1];
    @Positive
  }
