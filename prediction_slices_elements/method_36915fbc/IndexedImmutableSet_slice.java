// Source-based slice around line 58
// Method: <com.google.common.collect.IndexedImmutableSet: ImmutableList createAsList()>

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

      @Override
      boolean isPartialView() {
        return IndexedImmutableSet.this.isPartialView();
      }
