// Source-based slice around line 590
// Method: <com.google.common.io.MoreFiles: void deleteDirectoryContents(Path,RecursiveDeleteOption)>

   *
   * @throws NoSuchFileException if {@code path} does not exist <i>(optional specific exception)</i>
   * @throws NotDirectoryException if the file at {@code path} is not a directory <i>(optional
   *     specific exception)</i>
   * @throws InsecureRecursiveDeleteException if the security of recursive deletes can't be
   *     guaranteed for the file system and {@link RecursiveDeleteOption#ALLOW_INSECURE} was not
   *     specified
   * @throws IOException if one or more files can't be deleted for any reason
   */
  public static void deleteDirectoryContents(Path path, RecursiveDeleteOption... options)
      throws IOException {
    Collection<IOException> exceptions = null; // created lazily if needed
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
      if (stream instanceof SecureDirectoryStream) {
        SecureDirectoryStream<Path> sds = (SecureDirectoryStream<Path>) stream;
        exceptions = deleteDirectoryContentsSecure(sds);
      } else {
        checkAllowsInsecure(path, options);
        exceptions = deleteDirectoryContentsInsecure(stream);
      }
