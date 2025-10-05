// Source-based slice around line 672
// Method: <com.google.common.reflect.ClassPath: File toFile(URL)>


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
    } catch (URISyntaxException e) { // URL.toURI() doesn't escape chars.
      return new File(url.getPath()); // Accepts non-escaped chars like space.
    }
  }
}
