// Source-based slice around line 36
// Method: <com.google.common.collect.testing.Platform: void checkCast(Class,Object)>

 * @author Hayward Chan
 */
@GwtCompatible
final class Platform {
  static <T> T[] clone(T[] array) {
    return array.clone();
  }

  // Class.cast is not supported in GWT.  This method is a no-op in GWT.
  static void checkCast(Class<?> clazz, Object obj) {
    Object unused = clazz.cast(obj);
  }

  static String format(String template, Object... args) {
    return String.format(Locale.ROOT, template, args);
  }

  private Platform() {}
}
