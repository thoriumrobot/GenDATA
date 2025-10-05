// Source-based slice around line 113
// Method: <com.google.common.base.MoreObjects: ToStringHelper toStringHelper(Object)>

   *     .toString();
   * }
   *
   * <p>Note that in GWT, class names are often obfuscated.
   *
   * @param self the object to generate the string for (typically {@code this}), used only for its
   *     class name
   * @since 18.0 (since 2.0 as {@code Objects.toStringHelper()}).
   */
  public static ToStringHelper toStringHelper(Object self) {
    return new ToStringHelper(self.getClass().getSimpleName());
  }

  /**
   * Creates an instance of {@link ToStringHelper} in the same manner as {@link
   * #toStringHelper(Object)}, but using the simple name of {@code clazz} instead of using an
   * instance's {@link Object#getClass()}.
   *
   * <p>Note that in GWT, class names are often obfuscated.
   *
