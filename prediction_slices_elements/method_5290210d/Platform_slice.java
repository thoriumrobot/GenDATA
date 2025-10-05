// Source-based slice around line 40
// Method: <com.google.common.collect.testing.Platform: String format(String,Object)>

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
