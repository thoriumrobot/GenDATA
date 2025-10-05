// Source-based slice around line 155
// Method: <com.google.common.base.StandardSystemProperty: String value()>

   *
   * <p>Note that {@code StandardSystemProperty} does not provide constants for more recently added
   * properties, including:
   *
   * <ul>
   *   <li>{@code java.vendor.version} (added in Java 11, listed as optional as of Java 13)
   *   <li>{@code jdk.module.*} (added in Java 9, optional)
   * </ul>
   */
  public @Nullable String value() {
    return System.getProperty(key);
  }

  /** Returns a string representation of this system property. */
  @Override
  public String toString() {
    return key() + "=" + value();
  }
}
