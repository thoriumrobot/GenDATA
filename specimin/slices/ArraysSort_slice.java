  void sortInt(int @MinLen(10) [] nums) {
    // Checks the correct handling of the toIndex parameter
    Arrays.sort(nums, 0, 10);
    // :: error: (argument)
    Arrays.sort(nums, 0, 11);
  }
