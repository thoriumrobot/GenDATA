    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStreamReader;

    @Positive
public class SkipBufferedReader {
    @Positive
  public static void method() throws IOException {
    @Positive
    BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));

    // :: error: (argument)
    @Positive
    bufferedReader.skip(-1);

    @Positive
    bufferedReader.skip(1);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
