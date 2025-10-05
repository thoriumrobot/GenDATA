// Source-based slice around line 484
// Method: <com.google.common.base.Converter: boolean equals(Object)>

   *
   * <p><b>Warning: do not depend</b> on the behavior of this method.
   *
   * <p>Historically, {@code Converter} instances in this library have implemented this method to
   * recognize certain cases where distinct {@code Converter} instances would in fact behave
   * identically. However, this is not true of {@code Converter} implementations in general. It is
   * best not to depend on it.
   */
  @Override
  public boolean equals(@Nullable Object object) {
    return super.equals(object);
  }

  // Static converters

  /**
   * Returns a converter based on separate forward and backward functions. This is useful if the
   * function instances already exist, or so that you can supply lambda expressions. If those
   * circumstances don't apply, you probably don't need to use this; subclass {@code Converter} and
   * implement its {@link #doForward} and {@link #doBackward} methods directly.
