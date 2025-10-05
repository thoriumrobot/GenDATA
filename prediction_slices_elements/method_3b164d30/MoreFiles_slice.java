// Source-based slice around line 518
// Method: <com.google.common.io.MoreFiles: void deleteRecursively(Path,RecursiveDeleteOption)>

   * pass {@link RecursiveDeleteOption#ALLOW_INSECURE} to this method to override that behavior.
   *
   * @throws NoSuchFileException if {@code path} does not exist <i>(optional specific exception)</i>
   * @throws InsecureRecursiveDeleteException if the security of recursive deletes can't be
   *     guaranteed for the file system and {@link RecursiveDeleteOption#ALLOW_INSECURE} was not
   *     specified
   * @throws IOException if {@code path} or any file in the subtree rooted at it can't be deleted
   *     for any reason
   */
  public static void deleteRecursively(Path path, RecursiveDeleteOption... options)
      throws IOException {
    Path parentPath = getParentPath(path);
    if (parentPath == null) {
      throw new FileSystemException(path.toString(), null, "can't delete recursively");
    }

    Collection<IOException> exceptions = null; // created lazily if needed
    try {
      boolean sdsSupported = false;
      try (DirectoryStream<Path> parent = Files.newDirectoryStream(parentPath)) {
