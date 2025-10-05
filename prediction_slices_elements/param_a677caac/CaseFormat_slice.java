// Source-based slice around line 163
// Method: <com.google.common.base.CaseFormat: Converter converterTo(CaseFormat)>

        : requireNonNull(out).append(format.normalizeWord(s.substring(i))).toString();
  }

  /**
   * Returns a serializable {@code Converter} that converts strings from this format to {@code
   * targetFormat}.
   *
   * @since 16.0
   */
  public Converter<String, String> converterTo(CaseFormat targetFormat) {
    return new StringConverter(this, targetFormat);
  }

  private static final class StringConverter extends Converter<String, String>
      implements Serializable {

    private final CaseFormat sourceFormat;
    private final CaseFormat targetFormat;

    StringConverter(CaseFormat sourceFormat, CaseFormat targetFormat) {
