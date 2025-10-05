// Source-based slice around line 61
// Method: <com.google.common.collect.ImmutableMapEntry: K getKey()>

  ImmutableMapEntry(K key, V value) {
    super(key, value);
    checkEntryNotNull(key, value);
  }

  // Redeclare methods to make them `final`, just to be extra-safe.

  @Override
  @ParametricNullness
  public final K getKey() {
    return super.getKey();
  }

  @Override
  @ParametricNullness
  public final V getValue() {
    return super.getValue();
  }

  @Override
