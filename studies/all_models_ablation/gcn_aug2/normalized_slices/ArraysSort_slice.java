    @Positive
  void sortInt(int @MinLen(10) [] nums) {
    // Checks the correct handling of the toIndex parameter
    @Positive
    Arrays.sort(nums, 0, 10);
    // :: error: (argument)
    @Positive
    Arrays.sort(nums, 0, 11);
    @Positive
  }
