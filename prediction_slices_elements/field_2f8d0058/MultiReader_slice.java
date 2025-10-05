// Source-based slice around line 36
// Method: com.google.common.io.MultiReader.it

/**
 * A {@link Reader} that concatenates multiple readers.
 *
 * @author Bin Zhu
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
final class MultiReader extends Reader {
  private final Iterator<? extends CharSource> it;
  private @Nullable Reader current;

  MultiReader(Iterator<? extends CharSource> readers) throws IOException {
    this.it = readers;
    advance();
  }

  /** Closes the current reader and opens the next one, if any. */
  private void advance() throws IOException {
    close();
