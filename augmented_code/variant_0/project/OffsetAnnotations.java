/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.io.*;

    @Positive
public class OffsetAnnotations {
    @Positive
  public static void OffsetAnnotationsReader() throws IOException {
    @Positive
    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
    @Positive
    char[] buffer = new char[10];
    // :: error: (argument)
    @Positive
    bufferedReader.read(buffer, 5, 7);
    @Positive
  }

    @Positive
  public static void OffsetAnnotationsWriter() throws IOException {
    @Positive
    BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(System.out));
    @Positive
    char[] buffer = new char[10];
    // :: error: (argument)
    @Positive
    bufferedWriter.write(buffer, 5, 7);
    @Positive
  }
    @Positive
}
