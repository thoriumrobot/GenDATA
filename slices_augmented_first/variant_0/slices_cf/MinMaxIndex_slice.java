    @Positive
  void indexOrHigh(String str, @IndexOrHigh("#1") int i1, @IndexOrHigh("#1") int i2) {
    @Positive
    str.substring(Math.max(i1, i2));
    @Positive
    str.substring(Math.min(i1, i2));
    @Positive
  }
