    @Positive
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
    @Positive
    char[] arr = new char[size];
    @Positive
    arr[index] = val;
    @Positive
  }
