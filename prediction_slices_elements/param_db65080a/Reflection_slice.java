// Source-based slice around line 60
// Method: <com.google.common.reflect.Reflection: void initialize(Class)>

   * href="http://java.sun.com/docs/books/jls/third_edition/html/execution.html#12.4.2">JLS Section
   * 12.4.2</a>.
   *
   * <p>WARNING: Normally it's a smell if a class needs to be explicitly initialized, because static
   * state hurts system maintainability and testability. In cases when you have no choice while
   * interoperating with a legacy framework, this method helps to keep the code less ugly.
   *
   * @throws ExceptionInInitializerError if an exception is thrown during initialization of a class
   */
  public static void initialize(Class<?>... classes) {
    for (Class<?> clazz : classes) {
      try {
        Class.forName(clazz.getName(), true, clazz.getClassLoader());
      } catch (ClassNotFoundException e) {
        throw new AssertionError(e);
      }
    }
  }

  /**
