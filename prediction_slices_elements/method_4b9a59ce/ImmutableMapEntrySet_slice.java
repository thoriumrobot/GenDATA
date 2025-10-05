// Source-based slice around line 99
// Method: <com.google.common.collect.ImmutableMapEntrySet: int size()>

      return super.writeReplace();
    }
  }

  ImmutableMapEntrySet() {}

  abstract ImmutableMap<K, V> map();

  @Override
  public int size() {
    return map().size();
  }

  @Override
  public boolean contains(@Nullable Object object) {
    if (object instanceof Entry) {
      Entry<?, ?> entry = (Entry<?, ?>) object;
      V value = map().get(entry.getKey());
      return value != null && value.equals(entry.getValue());
    }
