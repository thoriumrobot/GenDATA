// Source-based slice around line 477
// Method: <com.google.common.io.MoreFiles: String getNameWithoutExtension(Path)>

    int dotIndex = fileName.lastIndexOf('.');
    return dotIndex == -1 ? "" : fileName.substring(dotIndex + 1);
  }

  /**
   * Returns the file name without its <a
   * href="http://en.wikipedia.org/wiki/Filename_extension">file extension</a> or path. This is
   * similar to the {@code basename} unix command. The result does not include the '{@code .}'.
   */
  public static String getNameWithoutExtension(Path path) {
    Path name = path.getFileName();

    // null for empty paths and root-only paths
    if (name == null) {
      return "";
    }

    String fileName = name.toString();
    int dotIndex = fileName.lastIndexOf('.');
    return dotIndex == -1 ? fileName : fileName.substring(0, dotIndex);
