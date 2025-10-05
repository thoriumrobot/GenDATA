// Source-based slice around line 150
// Method: com.google.common.base.Converter.reverse

 * Fortunately, if anyone does want to use a Converter as a `Function<@Nullable A, @Nullable B>`,
 * it's easy to get one: `converter::convert`.
 *
 * [*] In annotating this class, we're ignoring LegacyConverter.
 */
public abstract class Converter<A, B> implements Function<A, B> {
  private final boolean handleNullAutomatically;

  // We lazily cache the reverse view to avoid allocating on every call to reverse().
  @LazyInit @RetainedWith private transient @Nullable Converter<B, A> reverse;

  /** Constructor for use by subclasses. */
  protected Converter() {
    this(true);
  }

  /** Constructor used only by {@code LegacyConverter} to suspend automatic null-handling. */
  Converter(boolean handleNullAutomatically) {
    this.handleNullAutomatically = handleNullAutomatically;
  }
