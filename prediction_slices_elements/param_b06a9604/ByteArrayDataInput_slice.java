// Source-based slice around line 40
// Method: <com.google.common.io.ByteArrayDataInput: void readFully(byte[])>

 * of the supertype's contract, which specifies a checked exception.
 *
 * @author Kevin Bourrillion
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
public interface ByteArrayDataInput extends DataInput {
  @Override
  void readFully(byte[] b);

  @Override
  void readFully(byte[] b, int off, int len);

  // not guaranteed to skip n bytes so result should NOT be ignored
  // use ByteStreams.skipFully or one of the read methods instead
  @Override
  int skipBytes(int n);

  @CanIgnoreReturnValue // to skip a byte
