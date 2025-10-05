// Source-based slice around line 49
// Method: com.google.common.net.HostSpecifier.canonicalForm

 * InternetDomainName} rather than this class.
 *
 * @author Craig Berry
 * @since 5.0
 */
@J2ktIncompatible
@GwtIncompatible
public final class HostSpecifier {

  private final String canonicalForm;

  private HostSpecifier(String canonicalForm) {
    this.canonicalForm = canonicalForm;
  }

  /**
   * Returns a {@code HostSpecifier} built from the provided {@code specifier}, which is already
   * known to be valid. If the {@code specifier} might be invalid, use {@link #from(String)}
   * instead.
   *
