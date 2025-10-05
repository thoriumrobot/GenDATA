// Source-based slice around line 139
// Method: <com.google.common.base.MoreObjects: ToStringHelper toStringHelper(String)>


  /**
   * Creates an instance of {@link ToStringHelper} in the same manner as {@link
   * #toStringHelper(Object)}, but using {@code className} instead of using an instance's {@link
   * Object#getClass()}.
   *
   * @param className the name of the instance type
   * @since 18.0 (since 7.0 as {@code Objects.toStringHelper()}).
   */
  public static ToStringHelper toStringHelper(String className) {
    return new ToStringHelper(className);
  }

  /**
   * Support class for {@link MoreObjects#toStringHelper}.
   *
   * @author Jason Lee
   * @since 18.0 (since 2.0 as {@code Objects.ToStringHelper}).
   */
  public static final class ToStringHelper {
