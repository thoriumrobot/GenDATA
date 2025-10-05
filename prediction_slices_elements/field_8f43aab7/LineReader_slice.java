// Source-based slice around line 47
// Method: com.google.common.io.LineReader.lineBuf

@J2ktIncompatible
@GwtIncompatible
public final class LineReader {
  private final Readable readable;
  private final @Nullable Reader reader;
  private final CharBuffer cbuf = createBuffer();
  private final char[] buf = cbuf.array();

  private final Queue<String> lines = new ArrayDeque<>();
  private final LineBuffer lineBuf =
      new LineBuffer() {
        @Override
        protected void handleLine(String line, String end) {
          lines.add(line);
        }
      };

  /** Creates a new instance that will read lines from the given {@code Readable} object. */
  public LineReader(Readable readable) {
    this.readable = checkNotNull(readable);
