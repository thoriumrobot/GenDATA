// Source-based slice around line 619
// Method: <com.google.common.io.MoreFiles: Collection deleteRecursivelySecure(SecureDirectoryStream,Path)>

      throwDeleteFailed(path, exceptions);
    }
  }

  /**
   * Secure recursive delete using {@code SecureDirectoryStream}. Returns a collection of exceptions
   * that occurred or null if no exceptions were thrown.
   */
  private static @Nullable Collection<IOException> deleteRecursivelySecure(
      SecureDirectoryStream<Path> dir, Path path) {
    Collection<IOException> exceptions = null;
    try {
      if (isDirectory(dir, path, NOFOLLOW_LINKS)) {
        try (SecureDirectoryStream<Path> childDir = dir.newDirectoryStream(path, NOFOLLOW_LINKS)) {
          exceptions = deleteDirectoryContentsSecure(childDir);
        }

        // If exceptions is not null, something went wrong trying to delete the contents of the
        // directory, so we shouldn't try to delete the directory as it will probably fail.
        if (exceptions == null) {
