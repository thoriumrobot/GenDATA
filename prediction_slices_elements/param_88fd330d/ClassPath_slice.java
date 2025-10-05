// Source-based slice around line 123
// Method: <com.google.common.reflect.ClassPath: ClassPath from(ClassLoader)>

   *   <li>{@link URLClassLoader} instances' {@code file:} URLs
   *   <li>the {@linkplain ClassLoader#getSystemClassLoader() system class loader}. To search the
   *       system class loader even when it is not a {@link URLClassLoader} (as in Java 9), {@code
   *       ClassPath} searches the files from the {@code java.class.path} system property.
   * </ul>
   *
   * @throws IOException if the attempt to read class path resources (jar files or directories)
   *     failed.
   */
  public static ClassPath from(ClassLoader classloader) throws IOException {
    ImmutableSet<LocationInfo> locations = locationsFrom(classloader);

    // Add all locations to the scanned set so that in a classpath [jar1, jar2], where jar1 has a
    // manifest with Class-Path pointing to jar2, we won't scan jar2 twice.
    Set<File> scanned = new HashSet<>();
    for (LocationInfo location : locations) {
      scanned.add(location.file());
    }

    // Scan all locations
