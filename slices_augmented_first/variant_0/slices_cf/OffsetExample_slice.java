    @Positive
  void example1(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    int i = 0;
        while (i < y) {
            @Positive
      a[i + j] = 1;
    @Positive
            i++;
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
    int i = 0;
        while (i < y) {
            @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i + 0] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
            i++;
        }
    @Positive
  }
