// Source-based slice around line 69
// Method: com.google.common.io.FileBackedOutputStream.fileThreshold

 *
 * @author Chris Nokleberg
 * @since 1.0
 */
@Beta
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
