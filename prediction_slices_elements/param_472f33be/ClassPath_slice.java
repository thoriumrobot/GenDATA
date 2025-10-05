// Source-based slice around line 578
// Method: <com.google.common.reflect.ClassPath: ImmutableSet getClassPathFromManifest(File,Manifest)>


  /**
   * Returns the class path URIs specified by the {@code Class-Path} manifest attribute, according
   * to <a
   * href="http://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Main_Attributes">JAR
   * File Specification</a>. If {@code manifest} is null, it means the jar file has no manifest, and
   * an empty set will be returned.
   */
  @VisibleForTesting
  static ImmutableSet<File> getClassPathFromManifest(File jarFile, @Nullable Manifest manifest) {
    if (manifest == null) {
      return ImmutableSet.of();
    }
    ImmutableSet.Builder<File> builder = ImmutableSet.builder();
    String classpathAttribute =
        manifest.getMainAttributes().getValue(Attributes.Name.CLASS_PATH.toString());
    if (classpathAttribute != null) {
      for (String path : CLASS_PATH_ATTRIBUTE_SEPARATOR.split(classpathAttribute)) {
        URL url;
        try {
