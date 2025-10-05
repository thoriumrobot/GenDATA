// Source-based slice around line 53
// Method: <com.google.common.collect.IndexedImmutableSet: int copyIntoArray(Object[],int)>

    checkNotNull(consumer);
    int n = size();
    for (int i = 0; i < n; i++) {
      consumer.accept(get(i));
    }
  }

  @Override
  @GwtIncompatible
  int copyIntoArray(@Nullable Object[] dst, int offset) {
    return asList().copyIntoArray(dst, offset);
  }

  @Override
  ImmutableList<E> createAsList() {
    return new ImmutableAsList<E>() {
      @Override
      public E get(int index) {
        return IndexedImmutableSet.this.get(index);
      }
