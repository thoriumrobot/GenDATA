// Source-based slice around line 664
// Method: <com.google.common.io.MoreFiles: Collection deleteRecursivelyInsecure(Path)>

    } catch (DirectoryIteratorException e) {
      return addException(exceptions, e.getCause());
    }
  }

  /**
   * Insecure recursive delete for file systems that don't support {@code SecureDirectoryStream}.
   * Returns a collection of exceptions that occurred or null if no exceptions were thrown.
   */
  private static @Nullable Collection<IOException> deleteRecursivelyInsecure(Path path) {
    Collection<IOException> exceptions = null;
    try {
      if (Files.isDirectory(path, NOFOLLOW_LINKS)) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
          exceptions = deleteDirectoryContentsInsecure(stream);
        }
      }

      // If exceptions is not null, something went wrong trying to delete the contents of the
      // directory, so we shouldn't try to delete the directory as it will probably fail.
