// Source-based slice around line 31
// Method: <com.google.common.collect.testing.Platform: T[] clone(T[])>

/**
 * Methods factored out so that they can be emulated differently in GWT.
 *
 * <p>This class is emulated in GWT.
 *
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
