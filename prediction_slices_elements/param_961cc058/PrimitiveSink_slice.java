// Source-based slice around line 96
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putBoolean(boolean)>

  @CanIgnoreReturnValue
  PrimitiveSink putFloat(float f);

  /** Puts a double into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putDouble(double d);

  /** Puts a boolean into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putBoolean(boolean b);

  /** Puts a character into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putChar(char c);

  /**
   * Puts each 16-bit code unit from the {@link CharSequence} into this sink.
   *
   * <p><b>Warning:</b> This method will produce different output than most other languages do when
   * running on the equivalent input. For cross-language compatibility, use {@link #putString},
