// Source-based slice around line 786
// Method: <com.google.common.io.MoreFiles: void throwDeleteFailed(Path,Collection)>


  /**
   * Throws an exception indicating that one or more files couldn't be deleted when deleting {@code
   * path} or its contents.
   *
   * <p>If there is only one exception in the collection, and it is a {@link NoSuchFileException}
   * thrown because {@code path} itself didn't exist, then throws that exception. Otherwise, the
   * thrown exception contains all the exceptions in the given collection as suppressed exceptions.
   */
  private static void throwDeleteFailed(Path path, Collection<IOException> exceptions)
      throws FileSystemException {
    NoSuchFileException pathNotFound = pathNotFound(path, exceptions);
    if (pathNotFound != null) {
      throw pathNotFound;
    }
    // TODO(cgdecker): Should there be a custom exception type for this?
    // Also, should we try to include the Path of each file we may have failed to delete rather
    // than just the exceptions that occurred?
    FileSystemException deleteFailed =
        new FileSystemException(
