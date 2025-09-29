import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class SkipBufferedReader {

    public static void method() throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
        bufferedReader.skip(-1);
        bufferedReader.skip(1);
    }
}
