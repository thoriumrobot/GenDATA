// Source-based slice around line 383
// Method: <com.google.common.collect.MapMakerInternalMap: WeakValueReference unsetWeakValueReference()>

      extends InternalEntry<K, V, E> {}

  /** Marker interface for {@link InternalEntry} implementations for weak values. */
  interface WeakValueEntry<K, V, E extends InternalEntry<K, V, E>> extends InternalEntry<K, V, E> {
    /** Gets the weak value reference held by entry. */
    WeakValueReference<K, V, E> getValueReference();
  }

  @SuppressWarnings("unchecked") // impl never uses a parameter or returns any non-null value
  static <K, V, E extends InternalEntry<K, V, E>>
      WeakValueReference<K, V, E> unsetWeakValueReference() {
    return (WeakValueReference<K, V, E>) UNSET_WEAK_VALUE_REFERENCE;
  }

  /** Concrete implementation of {@link InternalEntry} for strong keys and strong values. */
  static class StrongKeyStrongValueEntry<K, V>
      extends AbstractStrongKeyEntry<K, V, StrongKeyStrongValueEntry<K, V>>
      implements StrongValueEntry<K, V, StrongKeyStrongValueEntry<K, V>> {
    private volatile @Nullable V value = null;

