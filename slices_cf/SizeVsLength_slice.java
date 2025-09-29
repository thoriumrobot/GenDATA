    @Positive
  public int[] getArray(@NonNegative int size) {
    @Positive
    int[] values = new int[size];
    @Positive
    for (int i = 0; i < size; i++) {
      values[i] = 22;
    }
    return values;
  }
