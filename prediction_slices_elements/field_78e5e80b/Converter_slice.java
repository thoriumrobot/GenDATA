// Source-based slice around line 147
// Method: com.google.common.base.Converter.handleNullAutomatically

 * fallout from trying to switch. I would be shocked if the switch would offer benefits to anywhere
 * near enough users to justify the costs.
 *
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
