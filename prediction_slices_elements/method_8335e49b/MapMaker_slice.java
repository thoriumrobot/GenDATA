// Source-based slice around line 206
// Method: <com.google.common.collect.MapMaker: MapMaker weakKeys()>

   * <p><b>Warning:</b> when this method is used, the resulting map will use identity ({@code ==})
   * comparison to determine equality of keys, which is a technical violation of the {@link Map}
   * specification, and may not be what you expect.
   *
   * @throws IllegalStateException if the key strength was already set
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
