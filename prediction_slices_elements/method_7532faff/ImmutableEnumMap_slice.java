// Source-based slice around line 113
// Method: <com.google.common.collect.ImmutableEnumMap: boolean isPartialView()>

    return CollectSpliterators.map(delegate.entrySet().spliterator(), Maps::unmodifiableEntry);
  }

  @Override
  public void forEach(BiConsumer<? super K, ? super V> action) {
    delegate.forEach(action);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  // All callers of the constructor are restricted to <K extends Enum<K>>.
  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return new EnumSerializedForm<>(delegate);
  }

