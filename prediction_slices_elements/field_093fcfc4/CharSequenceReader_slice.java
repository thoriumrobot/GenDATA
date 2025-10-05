// Source-based slice around line 43
// Method: com.google.common.io.CharSequenceReader.mark

 * @author Colin Decker
 */
// TODO(cgdecker): make this public? as a type, or a method in CharStreams?
@J2ktIncompatible
@GwtIncompatible
final class CharSequenceReader extends Reader {

  private @Nullable CharSequence seq;
  private int pos;
  private int mark;

  /** Creates a new reader wrapping the given character sequence. */
  public CharSequenceReader(CharSequence seq) {
    this.seq = checkNotNull(seq);
  }

  private void checkOpen() throws IOException {
    if (seq == null) {
      throw new IOException("reader closed");
    }
