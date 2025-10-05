// Source-based slice around line 237
// Method: <com.google.common.collect.MapMakerInternalMap: MapMakerInternalMap createWithDummyValues(MapMaker)>

   * optimized to saved memory. Since {@link MapMaker.Dummy} is a singleton, we don't need to store
   * any values at all. Because of this optimization, {@code build.getValueStrength()} must be
   * {@link Strength#STRONG}.
   *
   * <p>This method is intended to only be used by the internal implementation of {@link Interners},
   * since a map of dummy values is the exact use case there.
   */
  static <K>
      MapMakerInternalMap<K, Dummy, ? extends InternalEntry<K, Dummy, ?>, ?> createWithDummyValues(
          MapMaker builder) {
    if (builder.getKeyStrength() == Strength.STRONG
        && builder.getValueStrength() == Strength.STRONG) {
      return new MapMakerInternalMap<>(builder, StrongKeyDummyValueEntry.Helper.instance());
    }
    if (builder.getKeyStrength() == Strength.WEAK
        && builder.getValueStrength() == Strength.STRONG) {
      return new MapMakerInternalMap<>(builder, WeakKeyDummyValueEntry.Helper.instance());
    }
    if (builder.getValueStrength() == Strength.WEAK) {
      throw new IllegalArgumentException("Map cannot have both weak and dummy values");
