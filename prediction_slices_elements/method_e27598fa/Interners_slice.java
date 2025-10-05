// Source-based slice around line 114
// Method: <com.google.common.collect.Interners: Interner newWeakInterner()>

  }

  /**
   * Returns a new thread-safe interner which retains a weak reference to each instance it has
   * interned, and so does not prevent these instances from being garbage-collected. This most
   * likely does not perform as well as {@link #newStrongInterner}, but is the best alternative when
   * the memory usage of that implementation is unacceptable.
   */
  @GwtIncompatible("java.lang.ref.WeakReference")
  public static <E> Interner<E> newWeakInterner() {
    return newBuilder().weak().build();
  }

  @VisibleForTesting
  static final class InternerImpl<E> implements Interner<E> {
    // MapMaker is our friend, we know about this type
    @VisibleForTesting final MapMakerInternalMap<E, Dummy, ?, ?> map;

    private InternerImpl(MapMaker mapMaker) {
      this.map =
