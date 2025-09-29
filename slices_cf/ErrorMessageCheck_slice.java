    @Positive
  void method3(@NonNegative int size, @NonNegative int value) {
    this.size = size;
    @Positive
    this.vDown = new int[this.size];
    // :: error: (array.access.unsafe.high)
    vDown[1 + value] = 10;
  }
