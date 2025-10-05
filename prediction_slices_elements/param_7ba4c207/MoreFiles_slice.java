// Source-based slice around line 691
// Method: <com.google.common.io.MoreFiles: Collection deleteDirectoryContentsInsecure(DirectoryStream)>

    }
  }

  /**
   * Simple, insecure method for deleting the contents of a directory for file systems that don't
   * support {@code SecureDirectoryStream}. Returns a collection of exceptions that occurred or null
   * if no exceptions were thrown.
   */
  private static @Nullable Collection<IOException> deleteDirectoryContentsInsecure(
      DirectoryStream<Path> dir) {
    Collection<IOException> exceptions = null;
    try {
      for (Path entry : dir) {
        exceptions = concat(exceptions, deleteRecursivelyInsecure(entry));
      }

      return exceptions;
    } catch (DirectoryIteratorException e) {
      return addException(exceptions, e.getCause());
    }
