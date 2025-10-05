// Source-based slice around line 41
// Method: com.google.common.io.LineReader.readable

 * java.io.BufferedReader#readLine()} but for all {@link Readable} objects, not just instances of
 * {@link Reader}.
 *
 * @author Chris Nokleberg
 * @since 1.0
 */
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
