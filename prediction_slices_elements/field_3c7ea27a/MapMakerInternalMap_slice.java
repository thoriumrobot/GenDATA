// Source-based slice around line 1035
// Method: com.google.common.collect.MapMakerInternalMap.UNSET_WEAK_VALUE_REFERENCE

    public Object getValue() {
      throw new AssertionError();
    }
  }

  /**
   * A singleton {@link WeakValueReference} used to denote an unset value in an entry with weak
   * values.
   */
  static final WeakValueReference<Object, Object, DummyInternalEntry> UNSET_WEAK_VALUE_REFERENCE =
      new WeakValueReference<Object, Object, DummyInternalEntry>() {
        @Override
        public @Nullable DummyInternalEntry getEntry() {
          return null;
        }

        @Override
        public void clear() {}

        @Override
