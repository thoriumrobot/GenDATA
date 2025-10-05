// Source-based slice around line 646
// Method: <com.google.common.io.MoreFiles: Collection deleteDirectoryContentsSecure(SecureDirectoryStream)>

    } catch (IOException e) {
      return addException(exceptions, e);
    }
  }

  /**
   * Secure method for deleting the contents of a directory using {@code SecureDirectoryStream}.
   * Returns a collection of exceptions that occurred or null if no exceptions were thrown.
   */
  private static @Nullable Collection<IOException> deleteDirectoryContentsSecure(
      SecureDirectoryStream<Path> dir) {
    Collection<IOException> exceptions = null;
    try {
      for (Path path : dir) {
        exceptions = concat(exceptions, deleteRecursivelySecure(dir, path.getFileName()));
      }

      return exceptions;
    } catch (DirectoryIteratorException e) {
      return addException(exceptions, e.getCause());
