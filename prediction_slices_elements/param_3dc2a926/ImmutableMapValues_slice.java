// Source-based slice around line 72
// Method: <com.google.common.collect.ImmutableMapValues: boolean contains(Object)>

    };
  }

  @Override
  public Spliterator<V> spliterator() {
    return CollectSpliterators.map(map.entrySet().spliterator(), Entry::getValue);
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return object != null && Iterators.contains(iterator(), object);
  }

  @Override
  boolean isPartialView() {
    return true;
  }

  @Override
  public ImmutableList<V> asList() {
