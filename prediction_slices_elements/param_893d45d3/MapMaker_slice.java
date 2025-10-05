// Source-based slice around line 211
// Method: <com.google.common.collect.MapMaker: MapMaker setKeyStrength(Strength)>

   * @see WeakReference
   */
  @CanIgnoreReturnValue
  @GwtIncompatible // java.lang.ref.WeakReference
  public MapMaker weakKeys() {
    return setKeyStrength(Strength.WEAK);
  }

  @CanIgnoreReturnValue
  MapMaker setKeyStrength(Strength strength) {
    checkState(keyStrength == null, "Key strength was already set to %s", keyStrength);
    keyStrength = checkNotNull(strength);
    if (strength != Strength.STRONG) {
      // STRONG could be used during deserialization.
      useCustomMap = true;
    }
    return this;
  }

  Strength getKeyStrength() {
