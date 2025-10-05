// Source-based slice around line 637
// Method: <com.google.common.reflect.ClassPath: ImmutableList parseJavaClassPath()>

    }
    return ImmutableList.of();
  }

  /**
   * Returns the URLs in the class path specified by the {@code java.class.path} {@linkplain
   * System#getProperty system property}.
   */
  @VisibleForTesting // TODO(b/65488446): Make this a public API.
  static ImmutableList<URL> parseJavaClassPath() {
    ImmutableList.Builder<URL> urls = ImmutableList.builder();
    for (String entry : Splitter.on(PATH_SEPARATOR.value()).split(JAVA_CLASS_PATH.value())) {
      try {
        try {
          urls.add(new File(entry).toURI().toURL());
        } catch (SecurityException e) { // File.toURI checks to see if the file is a directory
          urls.add(new URL("file", null, new File(entry).getAbsolutePath()));
        }
      } catch (MalformedURLException e) {
        logger.log(WARNING, "malformed classpath entry: " + entry, e);
