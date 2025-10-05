// Source-based slice around line 665
// Method: <com.google.common.reflect.ClassPath: String getClassName(String)>

   * File Specification</a>. Even though the specification only talks about relative urls, absolute
   * urls are actually supported too (for example, in Maven surefire plugin).
   */
  @VisibleForTesting
  static URL getClassPathEntry(File jarFile, String path) throws MalformedURLException {
    return new URL(jarFile.toURI().toURL(), path);
  }

  @VisibleForTesting
  static String getClassName(String filename) {
    int classNameEnd = filename.length() - CLASS_FILE_NAME_EXTENSION.length();
    return filename.substring(0, classNameEnd).replace('/', '.');
  }

  // TODO(benyu): Try java.nio.file.Paths#get() when Guava drops JDK 6 support.
  @VisibleForTesting
  static File toFile(URL url) {
    checkArgument(url.getProtocol().equals("file"));
    try {
      return new File(url.toURI()); // Accepts escaped characters like %20.
