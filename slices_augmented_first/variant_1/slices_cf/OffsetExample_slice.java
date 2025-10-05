    @Positive
  void example1(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example2(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    a[y] = 0;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
    }
    @Positive
  }
