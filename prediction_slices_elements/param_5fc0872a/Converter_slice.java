// Source-based slice around line 172
// Method: <com.google.common.base.Converter: B doForward(A)>


  /**
   * Returns a representation of {@code a} as an instance of type {@code B}. If {@code a} cannot be
   * converted, an unchecked exception (such as {@link IllegalArgumentException}) should be thrown.
   *
   * @param a the instance to convert; will never be null
   * @return the converted instance; <b>must not</b> be null
   */
  @ForOverride
  protected abstract B doForward(A a);

  /**
   * Returns a representation of {@code b} as an instance of type {@code A}. If {@code b} cannot be
   * converted, an unchecked exception (such as {@link IllegalArgumentException}) should be thrown.
   *
   * @param b the instance to convert; will never be null
   * @return the converted instance; <b>must not</b> be null
   * @throws UnsupportedOperationException if backward conversion is not implemented; this should be
   *     very rare. Note that if backward conversion is not only unimplemented but
   *     unimplement<i>able</i> (for example, consider a {@code Converter<Chicken, ChickenNugget>}),
