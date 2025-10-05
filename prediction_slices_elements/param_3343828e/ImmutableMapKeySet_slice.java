// Source-based slice around line 64
// Method: <com.google.common.collect.ImmutableMapKeySet: K get(int)>

    return map.keySpliterator();
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return map.containsKey(object);
  }

  @Override
  K get(int index) {
    return map.entrySet().asList().get(index).getKey();
  }

  @Override
  public void forEach(Consumer<? super K> action) {
    checkNotNull(action);
    map.forEach((k, v) -> action.accept(k));
  }

  @Override
