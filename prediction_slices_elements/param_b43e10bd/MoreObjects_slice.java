// Source-based slice around line 127
// Method: <com.google.common.base.MoreObjects: ToStringHelper toStringHelper(Class)>

   * Creates an instance of {@link ToStringHelper} in the same manner as {@link
   * #toStringHelper(Object)}, but using the simple name of {@code clazz} instead of using an
   * instance's {@link Object#getClass()}.
   *
   * <p>Note that in GWT, class names are often obfuscated.
   *
   * @param clazz the {@link Class} of the instance
   * @since 18.0 (since 7.0 as {@code Objects.toStringHelper()}).
   */
  public static ToStringHelper toStringHelper(Class<?> clazz) {
    return new ToStringHelper(clazz.getSimpleName());
  }

  /**
   * Creates an instance of {@link ToStringHelper} in the same manner as {@link
   * #toStringHelper(Object)}, but using {@code className} instead of using an instance's {@link
   * Object#getClass()}.
   *
   * @param className the name of the instance type
   * @since 18.0 (since 7.0 as {@code Objects.toStringHelper()}).
