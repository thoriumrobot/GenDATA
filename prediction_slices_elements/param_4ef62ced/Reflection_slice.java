// Source-based slice around line 44
// Method: <com.google.common.reflect.Reflection: String getPackageName(String)>

  public static String getPackageName(Class<?> clazz) {
    return getPackageName(clazz.getName());
  }

  /**
   * Returns the package name of {@code classFullName} according to the Java Language Specification
   * (section 6.7). Unlike {@link Class#getPackage}, this method only parses the class name, without
   * attempting to define the {@link Package} and hence load files.
   */
  public static String getPackageName(String classFullName) {
    int lastDot = classFullName.lastIndexOf('.');
    return (lastDot < 0) ? "" : classFullName.substring(0, lastDot);
  }

  /**
   * Ensures that the given classes are initialized, as described in <a
   * href="http://java.sun.com/docs/books/jls/third_edition/html/execution.html#12.4.2">JLS Section
   * 12.4.2</a>.
   *
   * <p>WARNING: Normally it's a smell if a class needs to be explicitly initialized, because static
