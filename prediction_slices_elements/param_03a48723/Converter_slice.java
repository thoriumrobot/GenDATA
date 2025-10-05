// Source-based slice around line 187
// Method: <com.google.common.base.Converter: A doBackward(B)>

   * @param b the instance to convert; will never be null
   * @return the converted instance; <b>must not</b> be null
   * @throws UnsupportedOperationException if backward conversion is not implemented; this should be
   *     very rare. Note that if backward conversion is not only unimplemented but
   *     unimplement<i>able</i> (for example, consider a {@code Converter<Chicken, ChickenNugget>}),
   *     then this is not logically a {@code Converter} at all, and should just implement {@link
   *     Function}.
   */
  @ForOverride
  protected abstract A doBackward(B b);

  // API (consumer-side) methods

  /**
   * Returns a representation of {@code a} as an instance of type {@code B}.
   *
   * @return the converted value; is null <i>if and only if</i> {@code a} is null
   */
  public final @Nullable B convert(@Nullable A a) {
    return correctedDoForward(a);
