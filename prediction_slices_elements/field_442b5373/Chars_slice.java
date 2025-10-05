// Source-based slice around line 60
// Method: com.google.common.primitives.Chars.BYTES

public final class Chars {
  private Chars() {}

  /**
   * The number of bytes required to represent a primitive {@code char} value.
   *
   * <p>Prefer {@link Character#BYTES} instead.
   */
  // We don't use Character.BYTES here because it's not available under J2KT.
  public static final int BYTES = Character.SIZE / Byte.SIZE;

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link
   * Character#hashCode(char)}.
   *
   * @param value a primitive {@code char} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Character.hashCode(value)")
  @InlineMeValidationDisabled(
