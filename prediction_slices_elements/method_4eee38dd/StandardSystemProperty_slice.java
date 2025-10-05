// Source-based slice around line 161
// Method: <com.google.common.base.StandardSystemProperty: String toString()>

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
