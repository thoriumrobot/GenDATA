  void example2(int @MinLen(2) [] a) {
    int j = 2;
    @Positive
    int x = a.length;
    int y = x - j;
    a[y] = 0;
    for (int i = 0; i < y; i++) {
      a[i + j] = 1;
      a[j + i] = 1;
      a[i + 0] = 1;
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
      a[i + 2 + j] = 1;
    }
  }

  void example3(int @MinLen(2) [] a) {
    int j = 2;
    @Positive
    for (int i = 0; i < a.length - 2; i++) {
      a[i + j] = 1;
    }
  }

    @NonNegative
  void example4(int[] a, int offset) {
    @Positive
    int max_index = a.length - offset;
    @NonNegative
    for (int i = 0; i < max_index; i++) {
      a[i + offset] = 0;
    }
  }

    @NonNegative
  void example5(int[] a, int offset) {
    @Positive
    for (int i = 0; i < a.length - offset; i++) {
      a[i + offset] = 0;
    }
  }
