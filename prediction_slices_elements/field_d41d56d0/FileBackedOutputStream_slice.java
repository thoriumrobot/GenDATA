// Source-based slice around line 74
// Method: com.google.common.io.FileBackedOutputStream.out

@J2ktIncompatible
@GwtIncompatible
@J2ObjCIncompatible
public final class FileBackedOutputStream extends OutputStream {
  private final int fileThreshold;
  private final boolean resetOnFinalize;
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
