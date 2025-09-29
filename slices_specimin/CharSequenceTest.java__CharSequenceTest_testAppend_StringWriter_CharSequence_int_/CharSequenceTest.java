import java.io.IOException;
import java.io.StringWriter;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;
import org.checkerframework.common.value.qual.StringVal;

public class CharSequenceTest {

    void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
        app.append(cs, i, i);
        app.append(cs, 1, 2);
    }
}
