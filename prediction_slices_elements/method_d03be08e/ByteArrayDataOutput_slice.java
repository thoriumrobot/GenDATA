// Source-based slice around line 33
// Method: <com.google.common.io.ByteArrayDataOutput: void write(int)>

 * identical functionality but do not throw {@link IOException}.
 *
 * @author Jayaprabhakar Kadarkarai
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public interface ByteArrayDataOutput extends DataOutput {
  @Override
  void write(int b);

  @Override
  void write(byte[] b);

  @Override
  void write(byte[] b, int off, int len);

  @Override
  void writeBoolean(boolean v);

