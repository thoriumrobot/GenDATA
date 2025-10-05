// Source-based slice around line 660
// Method: <com.google.common.reflect.ClassPath: URL getClassPathEntry(File,String)>

  }

  /**
   * Returns the absolute uri of the Class-Path entry value as specified in <a
   * href="http://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Main_Attributes">JAR
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
