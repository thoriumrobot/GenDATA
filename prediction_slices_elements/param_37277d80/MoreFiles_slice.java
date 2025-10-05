// Source-based slice around line 459
// Method: <com.google.common.io.MoreFiles: String getFileExtension(Path)>

   * <p><b>Note:</b> This method simply returns everything after the last '{@code .}' in the file's
   * name as determined by {@link Path#getFileName}. It does not account for any filesystem-specific
   * behavior that the {@link Path} API does not already account for. For example, on NTFS it will
   * report {@code "txt"} as the extension for the filename {@code "foo.exe:.txt"} even though NTFS
   * will drop the {@code ":.txt"} part of the name when the file is actually created on the
   * filesystem due to NTFS's <a
   * href="https://learn.microsoft.com/en-us/archive/blogs/askcore/alternate-data-streams-in-ntfs">Alternate
   * Data Streams</a>.
   */
  public static String getFileExtension(Path path) {
    Path name = path.getFileName();

    // null for empty paths and root-only paths
    if (name == null) {
      return "";
    }

    String fileName = name.toString();
    int dotIndex = fileName.lastIndexOf('.');
    return dotIndex == -1 ? "" : fileName.substring(dotIndex + 1);
