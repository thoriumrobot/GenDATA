// Source-based slice around line 363
// Method: <com.google.common.io.MoreFiles: boolean equal(Path,Path)>

  }

  /**
   * Returns true if the files located by the given paths exist, are not directories, and contain
   * the same bytes.
   *
   * @throws IOException if an I/O error occurs
   * @since 22.0
   */
  public static boolean equal(Path path1, Path path2) throws IOException {
    checkNotNull(path1);
    checkNotNull(path2);
    if (Files.isSameFile(path1, path2)) {
      return true;
    }

    /*
     * Some operating systems may return zero as the length for files denoting system-dependent
     * entities such as devices or pipes, in which case we must fall back on comparing the bytes
     * directly.
