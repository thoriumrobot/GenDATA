// Source-based slice around line 604
// Method: <com.google.common.reflect.ClassPath: ImmutableMap getClassPathEntries(ClassLoader)>

        if (url.getProtocol().equals("file")) {
          builder.add(toFile(url));
        }
      }
    }
    return builder.build();
  }

  @VisibleForTesting
  static ImmutableMap<File, ClassLoader> getClassPathEntries(ClassLoader classloader) {
    LinkedHashMap<File, ClassLoader> entries = new LinkedHashMap<>();
    // Search parent first, since it's the order ClassLoader#loadClass() uses.
    ClassLoader parent = classloader.getParent();
    if (parent != null) {
      entries.putAll(getClassPathEntries(parent));
    }
    for (URL url : getClassLoaderUrls(classloader)) {
      if (url.getProtocol().equals("file")) {
        File file = toFile(url);
        if (!entries.containsKey(file)) {
