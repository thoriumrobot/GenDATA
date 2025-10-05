// Source-based slice around line 243
// Method: <com.google.common.collect.MapMaker: MapMaker weakValues()>

   * methods {@link Map#containsValue containsValue}, {@link ConcurrentMap#remove(Object, Object)
   * remove(Object, Object)} and {@link ConcurrentMap#replace(Object, Object, Object) replace(K, V,
   * V)}, and may not be what you expect.
   *
   * @throws IllegalStateException if the value strength was already set
   * @see WeakReference
   */
  @CanIgnoreReturnValue
  @GwtIncompatible // java.lang.ref.WeakReference
  public MapMaker weakValues() {
    return setValueStrength(Strength.WEAK);
  }

  /**
   * A dummy singleton value type used by {@link Interners}.
   *
   * <p>{@link MapMakerInternalMap} can optimize for memory usage in this case; see {@link
   * MapMakerInternalMap#createWithDummyValues}.
   */
  enum Dummy {
