/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.io.PrintWriter;

    @Positive
public class AnnotatedJDKTest {

    @Positive
  public void printWriterWrite(PrintWriter writer) {
    @Positive
    writer.write(-1);

    @Positive
    writer.write(8);
    @Positive
  }
    @Positive
}
