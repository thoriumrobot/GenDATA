// Source-based slice around line 103
// Method: <com.google.common.collect.Interners: Interner newStrongInterner()>

  public static InternerBuilder newBuilder() {
    return new InternerBuilder();
  }

  /**
   * Returns a new thread-safe interner which retains a strong reference to each instance it has
   * interned, thus preventing these instances from being garbage-collected. If this retention is
   * acceptable, this implementation may perform better than {@link #newWeakInterner}.
   */
  public static <E> Interner<E> newStrongInterner() {
    return newBuilder().strong().build();
  }

  /**
   * Returns a new thread-safe interner which retains a weak reference to each instance it has
   * interned, and so does not prevent these instances from being garbage-collected. This most
   * likely does not perform as well as {@link #newStrongInterner}, but is the best alternative when
   * the memory usage of that implementation is unacceptable.
   */
  @GwtIncompatible("java.lang.ref.WeakReference")
