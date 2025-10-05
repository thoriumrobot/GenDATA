// Source-based slice around line 417
// Method: <com.google.common.io.MoreFiles: void createParentDirectories(Path,FileAttribute)>


  /**
   * Creates any necessary but nonexistent parent directories of the specified path. Note that if
   * this operation fails, it may have succeeded in creating some (but not all) of the necessary
   * parent directories. The parent directory is created with the given {@code attrs}.
   *
   * @throws IOException if an I/O error occurs, or if any necessary but nonexistent parent
   *     directories of the specified file could not be created.
   */
  public static void createParentDirectories(Path path, FileAttribute<?>... attrs)
      throws IOException {
    // Interestingly, unlike File.getCanonicalFile(), Path/Files provides no way of getting the
    // canonical (absolute, normalized, symlinks resolved, etc.) form of a path to a nonexistent
    // file. getCanonicalFile() can at least get the canonical form of the part of the path which
    // actually exists and then append the normalized remainder of the path to that.
    Path normalizedAbsolutePath = path.toAbsolutePath().normalize();
    Path parent = normalizedAbsolutePath.getParent();
    if (parent == null) {
      // The given directory is a filesystem root. All zero of its ancestors exist. This doesn't
      // mean that the root itself exists -- consider x:\ on a Windows machine without such a
