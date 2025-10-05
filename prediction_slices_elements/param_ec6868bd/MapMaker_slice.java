// Source-based slice around line 258
// Method: <com.google.common.collect.MapMaker: MapMaker setValueStrength(Strength)>

   *
   * <p>{@link MapMakerInternalMap} can optimize for memory usage in this case; see {@link
   * MapMakerInternalMap#createWithDummyValues}.
   */
  enum Dummy {
    VALUE
  }

  @CanIgnoreReturnValue
  MapMaker setValueStrength(Strength strength) {
    checkState(valueStrength == null, "Value strength was already set to %s", valueStrength);
    valueStrength = checkNotNull(strength);
    if (strength != Strength.STRONG) {
      // STRONG could be used during deserialization.
      useCustomMap = true;
    }
    return this;
  }

  Strength getValueStrength() {
