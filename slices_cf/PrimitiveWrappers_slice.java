    @NonNegative
  void int_Integer_access_equivalent(@IndexFor("#3") Integer i, @IndexFor("#3") int j, int[] a) {
    a[i] = a[j];
  }
