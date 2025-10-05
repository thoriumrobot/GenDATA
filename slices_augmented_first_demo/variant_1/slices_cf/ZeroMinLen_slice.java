
    @Positive
  @IndexFor("nums2") int current_index2;

    @Positive
  void test() {
    @Positive
    current_index = 0;
    // :: error: (assignment)
    @Positive
    current_index2 = 0;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
