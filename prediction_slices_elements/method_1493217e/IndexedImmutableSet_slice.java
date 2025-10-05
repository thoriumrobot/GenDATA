// Source-based slice around line 43
// Method: <com.google.common.collect.IndexedImmutableSet: void forEach(Consumer)>

    return asList().iterator();
  }

  @Override
  public Spliterator<E> spliterator() {
    return CollectSpliterators.indexed(size(), SPLITERATOR_CHARACTERISTICS, this::get);
  }

  @Override
  public void forEach(Consumer<? super E> consumer) {
    checkNotNull(consumer);
    int n = size();
    for (int i = 0; i < n; i++) {
      consumer.accept(get(i));
    }
  }

  @Override
  @GwtIncompatible
  int copyIntoArray(@Nullable Object[] dst, int offset) {
