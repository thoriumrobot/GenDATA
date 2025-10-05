// Source-based slice around line 196
// Method: <com.google.common.base.Converter: B convert(A)>

  protected abstract A doBackward(B b);

  // API (consumer-side) methods

  /**
   * Returns a representation of {@code a} as an instance of type {@code B}.
   *
   * @return the converted value; is null <i>if and only if</i> {@code a} is null
   */
  public final @Nullable B convert(@Nullable A a) {
    return correctedDoForward(a);
  }

  @Nullable B correctedDoForward(@Nullable A a) {
    if (handleNullAutomatically) {
      // TODO(kevinb): we shouldn't be checking for a null result at runtime. Assert?
      return a == null ? null : checkNotNull(doForward(a));
    } else {
      return unsafeDoForward(a);
    }
