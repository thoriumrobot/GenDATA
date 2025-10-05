// Source-based slice around line 80
// Method: com.google.common.io.FileBackedOutputStream.file

  private final ByteSource source;

  @GuardedBy("this")
  private OutputStream out;

  @GuardedBy("this")
  private @Nullable MemoryOutput memory;

  @GuardedBy("this")
  private @Nullable File file;

  /** ByteArrayOutputStream that exposes its internals. */
  private static final class MemoryOutput extends ByteArrayOutputStream {
    byte[] getBuffer() {
      return buf;
    }

    int getCount() {
      return count;
    }
