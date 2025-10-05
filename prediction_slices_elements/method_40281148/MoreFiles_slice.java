// Source-based slice around line 294
// Method: <com.google.common.io.MoreFiles: Traverser fileTraverser()>

   * created by this traverser if an {@link IOException} is thrown by a call to {@link
   * #listFiles(Path)}.
   *
   * <p>Example: {@code MoreFiles.fileTraverser().depthFirstPreOrder(Paths.get("/"))} may return the
   * following paths: {@code ["/", "/etc", "/etc/config.txt", "/etc/fonts", "/home", "/home/alice",
   * ...]}
   *
   * @since 23.5
   */
  public static Traverser<Path> fileTraverser() {
    return Traverser.forTree(MoreFiles::fileTreeChildren);
  }

  private static Iterable<Path> fileTreeChildren(Path dir) {
    if (Files.isDirectory(dir, NOFOLLOW_LINKS)) {
      try {
        return listFiles(dir);
      } catch (IOException e) {
        // the exception thrown when iterating a DirectoryStream if an I/O exception occurs
        throw new DirectoryIteratorException(e);
